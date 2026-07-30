"""
library-agent - SQLite 数据库模块

提供结构化字段存储，替代纯 JSON 遍历过滤。
存储：uploader, upload_time, description, extra_description 等文本字段。
不存储：embedding 向量（仍放在 data.json 中）。

使用方式：
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
    """获取数据库单例，自动建表。"""
    return ImageDB(str(DB_PATH))


class ImageDB:
    """图片结构化数据库（SQLite）。

    表结构：
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

    索引：
        - idx_uploader ON images(uploader)
        - idx_upload_time ON images(upload_time)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """建表（如果不存在）。"""
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
        """插入一条图片记录。

        Args:
            image_info: 包含 file_name, file_path, thumbnail_path,
                       uploader, upload_time, description, extra_description 的字典。

        Returns:
            int: 插入后的行 ID。
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
        """获取所有图片记录（不含 embedding）。"""
        cursor = self.conn.execute("SELECT * FROM images ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]

    def get_image_by_id(self, image_id: int) -> dict | None:
        """根据 ID 获取单条记录。"""
        cursor = self.conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def search_by_ids(self, ids: list[int], filters: dict = None) -> list[dict]:
        """根据 ID 列表和可选过滤条件查询图片。

        用于向量召回后的精排阶段：先用向量相似度筛出候选 ID，
        再通过 SQL 做结构化过滤。

        Args:
            ids: 候选图片 ID 列表（向量召回结果）。
            filters: 可选过滤条件，支持：
                - uploader: 上传人精确匹配
                - date_from: 起始日期 (YYYY-MM-DD)
                - date_to: 截止日期 (YYYY-MM-DD)

        Returns:
            list[dict]: 匹配的图片记录列表。
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
        """返回图片总数。"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM images")
        return cursor.fetchone()[0]

    def close(self):
        """关闭数据库连接。"""
        self.conn.close()