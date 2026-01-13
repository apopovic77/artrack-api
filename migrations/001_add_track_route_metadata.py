"""
Migration: Add metadata_json column to track_routes table

This adds support for storing route-specific metadata like knowledge
for audio guides.

Run with: python -m migrations.001_add_track_route_metadata
"""

import sqlite3
import os

def migrate():
    # Use the same database path as production
    db_path = os.environ.get("DATABASE_PATH", "/var/lib/api-arkturian/artrack.db")

    # For local dev, use local database
    if not os.path.exists(db_path):
        db_path = "artrack.db"

    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if column exists
    cursor.execute("PRAGMA table_info(track_routes)")
    columns = [col[1] for col in cursor.fetchall()]

    if "metadata_json" not in columns:
        print("Adding metadata_json column to track_routes...")
        cursor.execute("ALTER TABLE track_routes ADD COLUMN metadata_json TEXT DEFAULT '{}'")
        conn.commit()
        print("Migration complete!")
    else:
        print("Column metadata_json already exists, skipping.")

    conn.close()

if __name__ == "__main__":
    migrate()
