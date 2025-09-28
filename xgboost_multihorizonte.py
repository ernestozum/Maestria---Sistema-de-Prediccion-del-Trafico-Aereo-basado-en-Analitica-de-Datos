import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna
from prophet import Prophet

# ==============================
# 1) CARGA DE DATOS
# ==============================
user = "root"
password = "root"
host = "localhost"
port = "3306"
database = "atc_flight_data"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
query = """
SELECT anio, mes, dia, hora, total_vuelos
FROM vuelos_por_hora
ORDER BY anio, mes, dia, hora
"""
df = pd.read_sql(query, engine)
engine.dispose()

# ==============================
# 2) PREPROCESAMIENTO
# ==============================
df = df.rename(columns={"anio": "year", "mes": "month", "dia": "day", "hora": "hour"})
df['fecha_hora'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df.set_index('fecha_hora', inplace=True)
df = df[['total_vuelos']]

# Outlier handling (IQR)
Q1, Q3 = df['total_vuelos'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df['total_vuelos'] = np.clip(df['total_vuelos'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# ==============================
# 3) PROPHET ENTRENADO SOLO CON 2023 (CPU)
# ==============================
df_2023 = df[df.index.year == 2023]
df_prophet = df_2023.reset_index()[['fecha_hora', 'total_vuelos']].rename(columns={'fecha_hora': 'ds', 'total_vuelos': 'y'})

prophet = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
prophet.fit(df_prophet)

# Predicción Prophet completa (2023 + 2024 bisiesto)
future = prophet.make_future_dataframe(periods=24*366, freq='H')
forecast = prophet.predict(future)
df_prophet_pred = forecast[['ds', 'yhat']].set_index('ds')

df['prophet_pred'] = df_prophet_pred['yhat']
df['residual'] = df['total_vuelos'] - df['prophet_pred']

# ==============================
# 4) FEATURES BÁSICAS
# ==============================
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
festivos = pd.to_datetime(["2023-01-01","2023-04-06","2023-12-25","2024-01-01","2024-04-06","2024-12-25"])
df['is_holiday'] = df.index.normalize().isin(festivos).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week']/7)
df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week']/7)

# ==============================
# 5) UTILIDADES XGBOOST (GPU con fallback)
# ==============================
def crear_lags(df_in, target_col='residual'):
    df_in = df_in.copy()
    df_in['lag_1h']  = df_in[target_col].shift(1)
    df_in['lag_6h']  = df_in[target_col].shift(6)
    df_in['lag_12h'] = df_in[target_col].shift(12)
    df_in['lag_24h'] = df_in[target_col].shift(24)
    df_in['rolling_6h']  = df_in[target_col].shift(1).rolling(6).mean()
    df_in['rolling_12h'] = df_in[target_col].shift(1).rolling(12).mean()
    df_in['rolling_24h'] = df_in[target_col].shift(1).rolling(24).mean()
    return df_in

def entrenar_modelo(df_train, features, target, usar_gpu=True):
    # Asegurar float32 (mejor para GPU)
    X_train_np = df_train[features].astype(np.float32)
    y_train_np = df_train[target].astype(np.float32)

    def objective(trial, use_gpu_flag):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 300, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            # XGBoost 2.x: GPU/CPU se controla con 'device'
            'tree_method': 'hist',
            'device': 'cuda' if use_gpu_flag else 'cpu',
            # opcional: limita hilos CPU cuando device='cpu'
            # 'n_jobs': 0,
        }
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=1)
        model.fit(X_train_np, y_train_np)
        preds = model.predict(X_train_np)
        return mean_squared_error(y_train_np, preds)

    # Intentar GPU; si falla, fallback a CPU
    try:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, True), n_trials=10)
        best_params = study.best_params
        best_params.update({'tree_method': 'hist', 'device': 'cuda'})
    except Exception as e:
        print("⚠️ GPU no disponible para XGBoost o fallo en GPU. Cambiando a CPU. Detalle:", repr(e))
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, False), n_trials=5)
        best_params = study.best_params
        best_params.update({'tree_method': 'hist', 'device': 'cpu'})

    # Entrenamiento final con los mejores hiperparámetros
    model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, verbosity=1)
    model.fit(X_train_np, y_train_np)

    # Verificación clara en consola
    print("⚙️ XGBoost params finales:", model.get_xgb_params())
    try:
        booster = model.get_booster()
        print("🔎 Booster attrs:", booster.attributes())
    except Exception:
        pass

    return model

# ==============================
# 6) ENTRENAR MODELOS BASE (+6h y +12h) y STACKING +24h
# ==============================
df = crear_lags(df)
df['target_6h']  = df['residual'].shift(-6)
df['target_12h'] = df['residual'].shift(-12)
df['target_24h'] = df['residual'].shift(-24)
df.dropna(inplace=True)

features_base = [
    'hour','day_of_week','month','is_weekend','is_holiday',
    'lag_1h','lag_6h','lag_12h','lag_24h',
    'rolling_6h','rolling_12h','rolling_24h',
    'hour_sin','hour_cos','dow_sin','dow_cos'
]

# Entrenar con 2023
df_train = df[df.index.year == 2023]

model_6h  = entrenar_modelo(df_train, features_base, 'target_6h',  usar_gpu=True)
model_12h = entrenar_modelo(df_train, features_base, 'target_12h', usar_gpu=True)

# Generar predicciones base como features en TODO el dataset (float32)
df_features_base_all = df[features_base].astype(np.float32)
df['pred_6h']  = model_6h.predict(df_features_base_all)
df['pred_12h'] = model_12h.predict(df_features_base_all)

# Refiltrar train para stacked
df_train = df[df.index.year == 2023]
features_stacked = features_base + ['pred_6h', 'pred_12h']

model_24h_stacked = entrenar_modelo(df_train, features_stacked, 'target_24h', usar_gpu=True)

# ==============================
# 7) EVALUACIÓN EN 2024 (STACKED +24h)
# ==============================
df_test = df[df.index.year == 2024].copy()
# Para alinear: el target_24h se refiere a y(t+24), así que quitamos las últimas 24 filas al hacer predict
X_test_stacked = df_test[features_stacked].astype(np.float32).iloc[:-24]
y_true_idx = df_test.index[24:]  # índices donde existe y(t+24)

# Residual predicho +24h
pred_resid_24h = model_24h_stacked.predict(X_test_stacked)

# Reconstrucción: y_hat_final = prophet + residual_pred
yhat_prophet_24h = df_prophet_pred.loc[y_true_idx, 'yhat'].values
final_pred = yhat_prophet_24h + pred_resid_24h

# Métricas vs real
y_true = df.loc[y_true_idx, 'total_vuelos'].values
mse_stacked = mean_squared_error(y_true, final_pred)
acc_stacked = 100 - (np.abs(y_true - final_pred) / y_true * 100)
acc_stacked = np.clip(acc_stacked, 0, 100)

print(f"\n📊 Modelo Stacked +24h -> MSE: {mse_stacked:.2f} | Precisión: {np.mean(acc_stacked):.2f}%")

# ==============================
# 8) GRÁFICO (Primera semana 2024)
# ==============================
plot_end = y_true_idx[0] + pd.Timedelta(hours=24*7-1)
mask = (y_true_idx >= y_true_idx[0]) & (y_true_idx <= plot_end)
idx_plot = y_true_idx[mask]

plt.figure(figsize=(18,6))
plt.plot(idx_plot, df.loc[idx_plot, 'total_vuelos'], label='Real', color='black')
plt.plot(idx_plot, final_pred[:len(idx_plot)], label='Predicción Stacked +24h', linestyle='--', color='red')
plt.title("Stacking: Predicción +24h vs Real (Primera semana 2024)")
plt.xlabel("Fecha y Hora")
plt.ylabel("Total de Vuelos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
