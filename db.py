"""
library-agent - SQLite _

_ JSON _
_uploader, upload_time, description, extra_description _
_embedding _ data.json _

_
    from db import get_db
    db = get_db()
    db.add_image(...)
    db.search_by_filters(...)
"""

import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "images.db"


def get_db() -> "ImageDB":
    """_"""
    return ImageDB(str(DB_PATH))


class ImageDB:
    """_SQLite_

    _
        images (
            id          INTEGER PRIMARY KEY,
            file_name   TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            thumbnail_path TEXT,
            uploader    TEXT NOT NULL DEFAULT 'test_user',
            upload_time TEXT NOT NULL,
            description TEXT,
            extra_description TEXT
        )

    _
        - idx_uploader ON images(uploader)
        - idx_upload_time ON images(upload_time)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """_"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                thumbnail_path TEXT,
                uploader TEXT NOT NULL DEFAULT 'test_user',
                upload_time TEXT NOT NULL,
                description TEXT,
                extra_description TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_uploader
                ON images(uploader);

            CREATE INDEX IF NOT EXISTS idx_upload_time
                ON images(upload_time);
        """)
        self.conn.commit()

    def add_image(self, image_info: dict) -> int:
        """_

        Args:
            image_info: _ file_name, file_path, thumbnail_path,
                       uploader, upload_time, description, extra_description _

        Returns:
            int: _ ID_
        """
        cursor = self.conn.execute(
            """INSERT INTO images
               (file_name, file_path, thumbnail_path, uploader,
                upload_time, description, extra_description)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                image_info["file_name"],
                image_info["file_path"],
                image_info.get("thumbnail_path", ""),
                image_info.get("uploader", "test_user"),
                image_info.get("upload_time", datetime.now().isoformat()),
                image_info.get("description", ""),
                image_info.get("extra_description", ""),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_all_images(self) -> list[dict]:
        """_ embedding_"""
        cursor = self.conn.execute("SELECT * FROM images ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]

    def get_image_by_id(self, image_id: int) -> dict | None:
        """_ ID _"""
        cursor = self.conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def search_by_ids(self, ids: list[int], filters: dict = None) -> list[dict]:
        """_ ID _

        _ ID_
        _ SQL _

        Args:
            ids: _ ID _
            filters: _
                - uploader: _
                - date_from: _ (YYYY-MM-DD)
                - date_to: _ (YYYY-MM-DD)

        Returns:
            list[dict]: _
        """
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        conditions = [f"id IN ({placeholders})"]
        params = list(ids)

        if filters:
            if "uploader" in filters:
                conditions.append("uploader = ?")
                params.append(filters["uploader"])
            if "date_from" in filters:
                conditions.append("upload_time >= ?")
                params.append(filters["date_from"])
            if "date_to" in filters:
                conditions.append("upload_time <= ?")
                params.append(filters["date_to"])

        sql = f"SELECT * FROM images WHERE {' AND '.join(conditions)} ORDER BY id"
        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """_"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM images")
        return cursor.fetchone()[0]

    def close(self):
        """_"""
        self.conn.close()