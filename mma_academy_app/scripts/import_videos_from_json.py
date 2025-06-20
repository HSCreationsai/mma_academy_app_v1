import sqlite3
import json
import os

# --- CONFIG ---
DB_PATH = "mma_academy.db"
JSON_PATH = os.path.join("data", "backups", "videos_20250515_200639.json")

# --- CONNECT ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# --- LOAD JSON ---
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"❌ JSON not found at: {JSON_PATH}")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    videos = json.load(f)

print(f"🎥 Importing {len(videos)} videos...")

# --- IMPORT ---
added_count = 0
for video in videos:
    title = video["title"]
    url = video["url"]
    uploaded_by = video.get("uploaded_by", "unknown")

    # Check for existing video with same title or URL
    cursor.execute("SELECT id FROM videos WHERE url = ?", (url,))
    if cursor.fetchone():
        print(f"⚠️ Skipped (already exists): {title}")
        continue

    try:
        cursor.execute("""
            INSERT INTO videos (title, url, uploaded_by)
            VALUES (?, ?, ?)
        """, (title, url, uploaded_by))
        added_count += 1
    except sqlite3.Error as e:
        print(f"❌ DB insert failed for {title}: {e}")

conn.commit()
conn.close()

print(f"✅ Done. {added_count} videos added.")
