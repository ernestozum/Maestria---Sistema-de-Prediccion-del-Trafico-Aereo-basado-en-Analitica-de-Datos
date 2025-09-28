import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from prophet import Prophet

# ==============================
# 1. CARGA DE DATOS
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
# 2. PREPROCESAMIENTO
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
# 3. PROPHET CON 2023 (MODELADO DE TENDENCIA Y ESTACIONALIDAD)
# ==============================
df_2023 = df[df.index.year == 2023]
df_prophet = df_2023.reset_index()[['fecha_hora', 'total_vuelos']].rename(columns={'fecha_hora': 'ds', 'total_vuelos': 'y'})

prophet = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
prophet.fit(df_prophet)

future = prophet.make_future_dataframe(periods=24*366, freq='H')
forecast = prophet.predict(future)
df_prophet_pred = forecast[['ds', 'yhat']].set_index('ds')

# Residuales
df['prophet_pred'] = df_prophet_pred['yhat']
df['residual'] = df['total_vuelos'] - df['prophet_pred']

# ==============================
# 4. CREAR FEATURES PARA LSTM
# ==============================
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
festivos = pd.to_datetime(["2023-01-01","2023-04-06","2023-12-25","2024-01-01","2024-04-06","2024-12-25"])
df['is_holiday'] = df.index.normalize().isin(festivos).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)

# Lags y rolling
df['lag_1h'] = df['residual'].shift(1)
df['lag_6h'] = df['residual'].shift(6)
df['lag_12h'] = df['residual'].shift(12)
df['lag_24h'] = df['residual'].shift(24)
df['rolling_6h'] = df['residual'].shift(1).rolling(6).mean()
df['rolling_12h'] = df['residual'].shift(1).rolling(12).mean()
df['rolling_24h'] = df['residual'].shift(1).rolling(24).mean()

# Objetivos futuros
df['target_6h'] = df['residual'].shift(-6)
df['target_12h'] = df['residual'].shift(-12)
df['target_24h'] = df['residual'].shift(-24)
df.dropna(inplace=True)

# ==============================
# 5. DIVISIÓN TRAIN/TEST
# ==============================
df_train = df[df.index.year == 2023]
df_test = df[df.index.year == 2024]

features_base = ['hour','day_of_week','month','is_weekend','is_holiday',
                 'hour_sin','hour_cos','dow_sin','dow_cos',
                 'lag_1h','lag_6h','lag_12h','lag_24h',
                 'rolling_6h','rolling_12h','rolling_24h']

# ==============================
# 6. FUNCIÓN PARA CREAR SECUENCIAS LSTM
# ==============================
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(df_train[features_base + ['residual']])
scaled_test = scaler.transform(df_test[features_base + ['residual']])

def create_sequences(data, target_idx, window_size=48):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :])
        y.append(data[i+window_size, target_idx])
    return np.array(X), np.array(y)

WINDOW_SIZE = 48
X_train, y_train = create_sequences(scaled_train, target_idx=-1, window_size=WINDOW_SIZE)
X_test, y_test = create_sequences(scaled_test, target_idx=-1, window_size=WINDOW_SIZE)

# ==============================
# 7. DEFINICIÓN DEL MODELO LSTM
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==============================
# 8. ENTRENAMIENTO DE LSTM SOBRE RESIDUALES
# ==============================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

model = LSTMModel(input_size=X_train.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

# ==============================
# 9. PREDICCIÓN SOBRE 2024 (STACKED LSTM + PROPHET)
# ==============================
model.eval()
with torch.no_grad():
    residual_preds = model(X_test_t).detach().numpy().flatten()

# Reconstrucción: Prophet + Residual
residual_index = df_test.index[WINDOW_SIZE:]
final_preds = df_prophet_pred.loc[residual_index, 'yhat'].values + residual_preds

y_real = df.loc[residual_index, 'total_vuelos']
mse = mean_squared_error(y_real, final_preds)
accuracy = 100 - (np.abs(y_real - final_preds) / y_real * 100)

print(f"\n📊 Modelo LSTM Residual + Prophet -> MSE: {mse:.2f} | Precisión: {np.mean(accuracy):.2f}%")

# ==============================
# 10. GRÁFICO COMPARATIVO
# ==============================
plt.figure(figsize=(18,6))
plt.plot(residual_index[:24*7], y_real[:24*7], label="Real", color="black")
plt.plot(residual_index[:24*7], final_preds[:24*7], label="LSTM + Prophet (Predicción)", linestyle="--", color="red")
plt.title("LSTM + Prophet: Predicción de Vuelos (Primera semana 2024)")
plt.xlabel("Fecha y Hora")
plt.ylabel("Total de Vuelos")
plt.legend()
plt.grid(True)
plt.show()
