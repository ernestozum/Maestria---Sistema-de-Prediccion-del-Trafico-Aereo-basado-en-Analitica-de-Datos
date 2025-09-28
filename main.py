from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mysql.connector
from collections import Counter
import os

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conexión a MySQL
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="atc_flight_data"
    )

# Archivos estáticos y templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory="templates")

# HTML principal
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API: Vuelos por hora
@app.get("/api/vuelos-por-hora/")
def vuelos_por_hora(year: int, month: int, day: int):
    try:
        file_pattern = f".HISTORICAL.{year:04d}.{month:02d}.{day:02d}"

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT fp.departure_time
            FROM flight_plans fp
            JOIN metadata m ON fp.metadata_id = m.id
            WHERE m.file_name LIKE %s
        """, (f"%{file_pattern}",))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        hours = []
        for row in rows:
            time_str = row['departure_time']
            if time_str and len(time_str) >= 2:
                try:
                    hour = int(time_str[:2])
                    if 0 <= hour <= 23:
                        hours.append(hour)
                except ValueError:
                    continue

        counter = Counter(hours)
        result = [{"hour": h, "flight_count": counter.get(h, 0)} for h in range(24)]
        return result

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API: Fechas disponibles
@app.get("/api/fechas-disponibles", response_class=JSONResponse)
async def fechas_disponibles():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT file_name FROM metadata")
    rows = cursor.fetchall()
    conn.close()

    years = set()
    months = set()
    days = set()

    for row in rows:
        try:
            file_name = row["file_name"]
            parts = file_name.split(".")
            year = int(parts[2])
            month = int(parts[3])
            day = int(parts[4])
            years.add(year)
            months.add(month)
            days.add(day)
        except (IndexError, ValueError):
            continue

    return {
        "years": sorted(years),
        "months": sorted(months),
        "days": sorted(days)
    }

# API: Vuelos por mes
@app.get("/api/vuelos-por-mes/", response_class=JSONResponse)
def vuelos_por_mes(year: int):
    try:
        file_pattern = f".HISTORICAL.{year:04d}."

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT m.file_name
            FROM flight_plans fp
            JOIN metadata m ON fp.metadata_id = m.id
            WHERE m.file_name LIKE %s
        """, (f"%{file_pattern}%",))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        month_counter = Counter()
        for row in rows:
            file_name = row['file_name']
            parts = file_name.split(".")
            if len(parts) >= 4:
                try:
                    month = int(parts[3])
                    if 1 <= month <= 12:
                        month_counter[month] += 1
                except ValueError:
                    continue

        result = [{"month": m, "flight_count": month_counter.get(m, 0)} for m in range(1, 13)]
        return result

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vuelos-por-dia-en-mes/")
def vuelos_por_dia_en_mes(year: int, month: int):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT 
                CAST(SUBSTRING_INDEX(m.file_name, '.', -1) AS UNSIGNED) AS day,
                COUNT(*) AS flight_count
            FROM flight_plans f
            JOIN metadata m ON f.metadata_id = m.id
            WHERE 
                SUBSTRING_INDEX(SUBSTRING_INDEX(m.file_name, '.', -3), '.', 1) = %s
                AND SUBSTRING_INDEX(SUBSTRING_INDEX(m.file_name, '.', -2), '.', 1) = %s
            GROUP BY day
            ORDER BY day;
        """, (str(year), f"{int(month):02d}"))

        data = cursor.fetchall()
        return data

    except Error as e:
        print("Error fetching vuelos por día en mes:", e)
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.get("/api/estadisticas-aerolineas")
def estadisticas_aerolineas():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT icao_prefix, 
               flight_count
        FROM airlines
        ORDER BY flight_count DESC
    """)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return JSONResponse(content=data)

@app.get("/api/aeropuertos-frecuentes")
def aeropuertos_frecuentes():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Aeropuertos de salida más frecuentes
    cursor.execute("""
        SELECT 
            airports.icao_code AS airport_name,
            COUNT(*) AS flight_count
        FROM flight_plans
        JOIN airports ON flight_plans.departure_airport_id = airports.id
        GROUP BY flight_plans.departure_airport_id
        ORDER BY flight_count DESC
        LIMIT 50;
    """)
    salidas = cursor.fetchall()

    # Aeropuertos de llegada más frecuentes
    cursor.execute("""
        SELECT 
            airports.icao_code AS airport_name,
            COUNT(*) AS flight_count
        FROM flight_plans
        JOIN airports ON flight_plans.arrival_airport_id = airports.id
        GROUP BY flight_plans.arrival_airport_id
        ORDER BY flight_count DESC
        LIMIT 50;
    """)
    llegadas = cursor.fetchall()

    cursor.close()
    conn.close()

    return {
        "salidas": salidas,
        "llegadas": llegadas
    }
