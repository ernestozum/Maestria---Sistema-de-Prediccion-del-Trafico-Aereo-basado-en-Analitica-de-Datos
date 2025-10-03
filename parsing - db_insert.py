import os
import re
import mysql.connector

from datetime import datetime

# --- CONFIGURACIÓN DE BASE DE DATOS ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'atc_flight_data'
}

# Conexión a MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor(buffered=True)

# --- FUNCIONES ---

def get_or_create_airport_id(cursor, icao_code, name=None):
    cursor.execute("SELECT id FROM airports WHERE icao_code = %s", (icao_code,))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute(
        "INSERT INTO airports (icao_code, name) VALUES (%s, %s)",
        (icao_code, name)
    )
    return cursor.lastrowid

def insert_metadata_and_get_id(cursor, file_name, fpl_count):
    cursor.execute("""
        INSERT INTO metadata (file_name, fpl_count)
        VALUES (%s, %s)
    """, (file_name, fpl_count))
    return cursor.lastrowid

def insert_fpl_message(cursor, callsign, departure_airport_id, arrival_airport_id, departure_time, arrival_time, raw_route, metadata_id):
    cursor.execute("""
        INSERT INTO flight_plans (
            callsign,
            departure_airport_id,
            arrival_airport_id,
            departure_time,
            arrival_time,
            raw_route,
            metadata_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        callsign,
        departure_airport_id,
        arrival_airport_id,
        departure_time,
        arrival_time,
        raw_route,
        metadata_id
    ))
    return cursor.lastrowid

def insert_waypoints(cursor, fpl_message_id, waypoints):
    for index, name in enumerate(waypoints, 1):
        cursor.execute("""
            INSERT INTO waypoints (fpl_message_id, sequence, name)
            VALUES (%s, %s, %s)
        """, (fpl_message_id, index, name))


def insert_metadata(cursor, file_name, fpl_count):
    cursor.execute("""
        INSERT INTO metadata (file_name, fpl_count)
        VALUES (%s, %s)
    """, (file_name, fpl_count))

def is_file_processed(cursor, file_name):
    cursor.execute("SELECT id FROM metadata WHERE file_name = %s", (file_name,))
    return cursor.fetchone() is not None

def parse_waypoints(raw_route):
    tokens = re.split(r'\s+', raw_route.strip())
    waypoints = []

    for token in tokens:
        if "/" in token:
            point, suffix = token.split("/", 1)
            if not re.fullmatch(r"[NMFS]\d{3,4}F\d{3}", point):
                waypoints.append(point)
        elif re.fullmatch(r"[NMFS]\d{3,4}F\d{3}", token):
            continue
        else:
            waypoints.append(token)

    return waypoints

# --- PARSEO DE ARCHIVOS ---

folder_path = r"D:\Maestria\Proyecto_Final\Historical_Data"

fpl_pattern = re.compile(r"\(FPL-[^)]+\)", re.DOTALL)
callsign_pattern = re.compile(r"\(FPL-([A-Z0-9]+)")
departure_info_pattern = re.compile(r"-([A-Z]{4})(\d{4})")
route_pattern = re.compile(r"-[A-Z]\d{3,5}[A-Z]\d{3,5} (.+?)(?=\n-[A-Z]{3,5}\d{3,5})", re.DOTALL)
arrival_info_pattern = re.compile(r"\n-([A-Z]{3,5})(\d{3,5})")

for filename in os.listdir(folder_path):
    if filename.startswith(".HISTORICAL.2024"):
        file_path = os.path.join(folder_path, filename)
        
        # ✅ Verifica si ya fue procesado
        if is_file_processed(cursor, filename):
            print(f"⚠️ Archivo ya procesado previamente: {filename}")
            continue
        
        print(f"\n📄 Procesando archivo: {filename}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

            fpl_matches = fpl_pattern.findall(content)
            
            metadata_id = insert_metadata_and_get_id(cursor, filename, len(fpl_matches))
            
            print(f"🧾 Planes de vuelo encontrados: {len(fpl_matches)}")

            for fpl_msg in fpl_matches:
                # Callsign
                callsign_match = callsign_pattern.search(fpl_msg)
                callsign = callsign_match.group(1) if callsign_match else "N/A"

                # Departure
                departure_match = departure_info_pattern.search(fpl_msg)
                departure_airport = departure_match.group(1) if departure_match else "N/A"
                departure_time = departure_match.group(2) if departure_match else "N/A"

                # Raw route
                route_match = route_pattern.search(fpl_msg)
                raw_route = route_match.group(1).replace("\n", " ").strip() if route_match else "N/A"

                # Arrival
                arrival_match = arrival_info_pattern.search(fpl_msg, route_match.end() if route_match else 0)
                arrival_airport = arrival_match.group(1) if arrival_match else "N/A"
                arrival_time = arrival_match.group(2) if arrival_match else "N/A"

                
                print(f"✈️ Callsign: {callsign} | 🛫 {departure_airport} ⏰ {departure_time} | 🛬 {arrival_airport} ⏰ {arrival_time}")

                # Guardar aeropuertos si están presentes
                if departure_airport != "N/A":
                    departure_airport_id = get_or_create_airport_id(cursor, departure_airport)
                else:
                    departure_airport_id = None

                if arrival_airport != "N/A":
                    arrival_airport_id = get_or_create_airport_id(cursor, arrival_airport)
                else:
                    arrival_airport_id = None
                
                print(f"🔗 Aeropuertos insertados/verificados: {departure_airport_id=} {arrival_airport_id=}")


                fpl_message_id = insert_fpl_message(
                    cursor,
                    callsign,
                    departure_airport_id,
                    arrival_airport_id,
                    departure_time,
                    arrival_time,
                    raw_route,
                    metadata_id
                )

                print(f"📥 Mensaje FPL insertado con ID: {fpl_message_id}")                

                # Si la ruta es válida, parseamos los puntos
                if raw_route != "N/A":
                    waypoints = parse_waypoints(raw_route)
                    insert_waypoints(cursor, fpl_message_id, waypoints)
                    print(f"🧭 {len(waypoints)} waypoints insertados para {callsign}")
                else:
                    print("⚠️ Ruta no válida. No se insertaron waypoints.")                 
                
                
        # Confirmar todos los cambios para este archivo

        print(f"🗃️ Metadata registrada para archivo: {filename}")
        
        # Confirmar todos los cambios para este archivo
        conn.commit()

# Cerrar conexión
cursor.close()
conn.close()
