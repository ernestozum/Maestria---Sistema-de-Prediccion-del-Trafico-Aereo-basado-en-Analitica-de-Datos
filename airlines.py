import mysql.connector
import re
from collections import defaultdict

def extract_airline_prefix(callsign):
    """Extrae prefijo ICAO si es válido, o devuelve 'XAA' si es privado/desconocido."""
    if callsign is None:
        return "XAA"
    match = re.match(r"^([A-Z]{3})(\d+)$", callsign.strip())
    if match:
        return match.group(1)
    else:
        return "TBD"

def generar_tabla_airlines():
    # Conexión
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="atc_flight_data"
    )
    cursor = conn.cursor()


    # Obtener todos los callsigns
    cursor.execute("SELECT callsign FROM flight_plans")
    callsigns = cursor.fetchall()

    # Contar vuelos por prefijo
    prefix_counter = defaultdict(int)
    for (callsign,) in callsigns:
        prefix = extract_airline_prefix(callsign)
        prefix_counter[prefix] += 1

    # Construir lista de tuplas para insertar
    data_to_insert = []
    for prefix, count in prefix_counter.items():
        airline_name = "Privado/Desconocido" if prefix == "TBD" else None
        data_to_insert.append((prefix, airline_name, count))

    # Insertar o actualizar
    insert_query = """
        INSERT INTO airlines (icao_prefix, airline_name, flight_count)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            airline_name = VALUES(airline_name),
            flight_count = VALUES(flight_count)
    """
    cursor.executemany(insert_query, data_to_insert)

    # Confirmar
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Insertados/actualizados {len(data_to_insert)} registros en la tabla airlines.")

if __name__ == "__main__":
    generar_tabla_airlines()
