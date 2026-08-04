"""
supabase_client — JD Cloud PostgreSQL + OSS storage module
Replaces Supabase REST API with direct psycopg2 + boto3 (S3-compatible).
"""

import os, io, hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
import boto3
from dotenv import load_dotenv

load_dotenv()

# ── DB config ──
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "library")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "Importantthings1")

# ── OSS config (S3-compatible) ──
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "https://s3.cn-north-1.jdcloud-oss.com")
OSS_ACCESS_KEY = os.getenv("OSS_ACCESS_KEY", "JDC_78A5C48965E8C5EDECF5882A1E1D")
OSS_SECRET_KEY = os.getenv("OSS_SECRET_KEY", "0792A10A3F5C8488BCC4A3B7DD6BB428")
OSS_BUCKET = os.getenv("OSS_BUCKET", "library-assets")


def _get_db():
    """Get a psycopg2 connection."""
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


def _get_oss():
    """Get boto3 S3 client for JD Cloud OSS."""
    from botocore.config import Config
    import os as _os
    _os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "WHEN_REQUIRED"
    return boto3.client(
        "s3",
        endpoint_url=OSS_ENDPOINT,
        aws_access_key_id=OSS_ACCESS_KEY,
        aws_secret_access_key=OSS_SECRET_KEY,
        region_name="cn-north-1",
        config=Config(signature_version="s3v4", request_checksum_calculation="when_required"),
    )


# ══════════════════════════════════════════
#  Database operations
# ══════════════════════════════════════════

def add_image(image_info: dict) -> dict:
    """Insert image record with embedding."""
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO images (file_name, file_path, thumbnail_path, uploader,
                            upload_time, description, extra_description,
                            source_file, source_url, embedding)
        VALUES (%(file_name)s, %(file_path)s, %(thumbnail_path)s, %(uploader)s,
                %(upload_time)s, %(description)s, %(extra_description)s,
                %(source_file)s, %(source_url)s, %(embedding)s)
        RETURNING id
    """, {
        "file_name": image_info["file_name"],
        "file_path": image_info["file_path"],
        "thumbnail_path": image_info.get("thumbnail_path", ""),
        "uploader": image_info.get("uploader", "test_user"),
        "upload_time": image_info.get("upload_time", datetime.now().isoformat()),
        "description": image_info.get("description", ""),
        "extra_description": image_info.get("extra_description", ""),
        "source_file": image_info.get("source_file", ""),
        "source_url": image_info.get("source_url", ""),
        "embedding": image_info["embedding"],
    })
    image_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    image_info["id"] = image_id
    return image_info


def get_all_images() -> list[dict]:
    """Fetch all image records (no embedding column)."""
    conn = _get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT id, file_name, file_path, thumbnail_path, uploader,
               upload_time, description, extra_description,
               source_file, source_url, duplicate_of
        FROM images ORDER BY id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if hasattr(v, 'isoformat'):
                d[k] = v.isoformat()
        result.append(d)
    return result


def get_image_by_id(image_id: int) -> Optional[dict]:
    """Get single image by ID."""
    conn = _get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM images WHERE id = %s", (image_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def search_similar(
    query_embedding: list[float],
    top_n: int = 3,
    match_threshold: float = 0.0,
    filters: Optional[dict] = None,
) -> list[dict]:
    """Vector similarity search via match_images RPC."""
    conn = _get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT * FROM match_images(%s::vector, %s, %s, %s, %s, %s)
    """, (
        query_embedding, match_threshold, top_n,
        filters.get("uploader") if filters else None,
        filters.get("date_from") if filters else None,
        filters.get("date_to") if filters else None,
    ))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if hasattr(v, 'isoformat'):
                d[k] = v.isoformat()
        result.append(d)
    return result


def count_images() -> int:
    """Return total image count."""
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM images")
    cnt = cur.fetchone()[0]
    cur.close()
    conn.close()
    return cnt


def clear_all() -> int:
    """Delete all images."""
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM images")
    conn.commit()
    deleted = cur.rowcount
    cur.close()
    conn.close()
    return deleted


# ══════════════════════════════════════════
#  Storage operations (JD Cloud OSS)
# ══════════════════════════════════════════

def upload_to_storage(bucket: str, local_path: Path, remote_name: Optional[str] = None) -> str:
    """Upload file to JD Cloud OSS, return public URL."""
    s3 = _get_oss()
    if remote_name is None:
        remote_name = _safe_storage_name(local_path.name)
    actual_bucket = OSS_BUCKET  # Use single bucket with prefix
    key = f"{bucket}/{remote_name}"
    data = local_path.read_bytes()
    for attempt in range(3):
        try:
            s3.put_object(Bucket=actual_bucket, Key=key, Body=data, ACL="public-read")
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  OSS retry {attempt+1}: {e}")
            import time as _t; _t.sleep(2)
    return f"{OSS_ENDPOINT}/{actual_bucket}/{key}"


def delete_from_storage(bucket: str, remote_name: str) -> None:
    """Delete file from OSS."""
    s3 = _get_oss()
    key = f"{bucket}/{remote_name}"
    s3.delete_object(Bucket=OSS_BUCKET, Key=key)


def _safe_storage_name(filename: str) -> str:
    """Sanitize filename for storage."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    if all(ord(c) < 128 for c in filename):
        return filename
    name_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    ascii_part = "".join(c for c in stem if c.isascii() and (c.isalnum() or c in "-_."))[:20]
    return f"{ascii_part}_{name_hash}{suffix}" if ascii_part else f"img_{name_hash}{suffix}"


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def check_connection() -> dict:
    try:
        cnt = count_images()
        return {"status": "ok", "image_count": cnt}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_supabase():
    """Backward compat: some old code imports this."""
    return None
