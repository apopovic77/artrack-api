import os
import sqlite3
from pathlib import Path
import piexif
import json
import subprocess
from storage.service import generic_storage, _get_gps_from_exif

DATABASE_PATH = os.getenv("ARTRACK_DATABASE_URL", "sqlite:////var/lib/api-arkturian/artrack.db").replace("sqlite:///", "/")

def get_gps_from_mp4(file_path: Path) -> tuple[float, float] | None:
    # ... (function remains the same)
    try:
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(file_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        
        location_str = tags.get("location") or tags.get("location-eng") or tags.get("com.apple.quicktime.location.ISO6709")
        
        if location_str:
            location_str = location_str.strip('/')
            if location_str.startswith('+'):
                location_str = location_str[1:]
            
            parts = location_str.replace('-', ' -').replace('+', ' ').split()
            if len(parts) >= 2:
                lat = float(parts[0])
                lon = float(parts[1])
                return lat, lon
    except Exception:
        pass
    return None

def backfill_gps_data_sql():
    print(f"Connecting to database at {DATABASE_PATH}...")
    try:
        con = sqlite3.connect(DATABASE_PATH)
        cur = con.cursor()

        print("Querying for image and video objects without GPS data...")
        cur.execute("""
            SELECT id, object_key, original_filename, mime_type
            FROM storage_objects
            WHERE (mime_type LIKE 'image/%' OR mime_type = 'video/mp4')
              AND latitude IS NULL
        """)
        objects_to_process = cur.fetchall()

        if not objects_to_process:
            print("No objects found that need GPS backfilling.")
            return

        print(f"Found {len(objects_to_process)} objects to process.")
        updated_count = 0

        for obj_id, object_key, original_filename, mime_type in objects_to_process:
            file_path = generic_storage.absolute_path_for_key(object_key)
            if not file_path.exists():
                print(f"  - SKIPPING: File not found for object {obj_id} at {file_path}")
                continue

            print(f"  - Processing {obj_id}: {original_filename}...", end="")
            
            gps_data = None
            try:
                if mime_type.startswith('image/'):
                    exif_data = piexif.load(str(file_path))
                    gps_data = _get_gps_from_exif(exif_data)
                elif mime_type == 'video/mp4':
                    gps_data = get_gps_from_mp4(file_path)

                if gps_data:
                    cur.execute(
                        "UPDATE storage_objects SET latitude = ?, longitude = ? WHERE id = ?",
                        (gps_data[0], gps_data[1], obj_id)
                    )
                    con.commit()
                    updated_count += 1
                    print(f" FOUND GPS -> Updated ({gps_data[0]:.4f}, {gps_data[1]:.4f})")
                else:
                    print(" No GPS found.")

            except Exception as e:
                print(f" ERROR: {e}")

        print(f"\nBackfill complete. Updated {updated_count} objects with GPS data.")

    except Exception as e:
        print(f"A database error occurred: {e}")
    finally:
        if 'con' in locals() and con:
            con.close()

if __name__ == "__main__":
    backfill_gps_data_sql()
