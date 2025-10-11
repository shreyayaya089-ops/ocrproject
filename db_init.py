import os
import sqlite3

print(os.path.abspath("vehicle_data.db"))

def init_db():
    conn = sqlite3.connect("vehicle_data.db")
    cur = conn.cursor()

    # Vehicles table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicles (
        plate TEXT PRIMARY KEY,
        owner_name TEXT,
        vehicle_type TEXT,
        registration_date TEXT
    );
    """)

    # Stolen vehicles
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stolen_vehicles (
        plate TEXT PRIMARY KEY
    );
    """)

    # Challans
    cur.execute("""
    CREATE TABLE IF NOT EXISTS challans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT,
        amount INTEGER,
        reason TEXT
    );
    """)

    # Scans
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        raw_text TEXT,
        detected_plate TEXT,
        corrected_plate TEXT,
        valid INTEGER DEFAULT 0,
        matched_vehicle_id INTEGER,
        user_corrected INTEGER DEFAULT 0,
        is_blacklisted INTEGER DEFAULT 0
    );
    """)

    # Entry/Exit simulation
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT,
        entry_time TIMESTAMP,
        exit_time TIMESTAMP,
        duration_seconds INTEGER
    );
    """)

    # Sample data
    vehicles = [
        ("MH12AB1234", "Ajay Sharma", "Car", "2021-04-12"),
        ("DL09BB4555", "Sneha Patel", "Truck", "2020-11-02"),
        ("KA03MN4321", "Ravi Kumar", "Bike", "2019-08-25"),
        ("RJ14CV0002", "Bablu Singh", "Bike", "2025-10-05")
    ]
    stolen = [("DL09BB4555",)]
    challans = [
        (None, "MH12AB1234", 500, "No Helmet"),
        (None, "KA03MN4321", 2000, "Speeding")
    ]

    cur.executemany("INSERT OR REPLACE INTO vehicles VALUES (?, ?, ?, ?)", vehicles)
    cur.executemany("INSERT OR REPLACE INTO stolen_vehicles VALUES (?)", stolen)
    cur.executemany("INSERT INTO challans VALUES (?, ?, ?, ?)", challans)

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# Corrected main block
if __name__ == "__main__":
    init_db()