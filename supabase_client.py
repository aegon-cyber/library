"""
supabase_client - Supabase _ + _

_ data.json + images.db + uploads/thumbnails/ _
_ Supabase REST API _ PostgreSQL_

_images (_ pgvector embedding _)
_uploads / thumbnails
_match_images (RPC)
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Monkey-patch httpx to use UTF-8 for header encoding (Render Linux fix)
import httpx as _httpx
_orig_normalize = _httpx._models._normalize_header_value
def _utf8_normalize(value, encoding=None):
    return _orig_normalize(value, encoding="utf-8")
_httpx._models._normalize_header_value = _utf8_normalize

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    """_ Supabase _"""
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


# ============================================================
# [CN]
# ============================================================

def add_image(image_info: dict) -> dict:
    """_ Supabase _

    Args:
        image_info: _ file_name, file_path, thumbnail_path, uploader,
                    upload_time, description, extra_description, embedding

    Returns:
        dict: _ id_
    """
    supabase = get_supabase()

    row = {
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
    }

    result = supabase.table("images").insert(row).execute()
    return dict(result.data[0]) if result.data else {}


def get_all_images() -> list[dict]:
    """Fetch all images via raw REST."""
    import httpx
    base = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    cols = "id,file_name,file_path,thumbnail_path,uploader,upload_time,description,extra_description,source_file,source_url,duplicate_of"
    r = httpx.get(
        f"{base}/rest/v1/images",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        params={"select": cols, "order": "id.asc"},
    )
    r.raise_for_status()
    return [dict(item) for item in r.json()]


def get_image_by_id(image_id: int) -> Optional[dict]:
    """_ ID _"""
    supabase = get_supabase()
    result = supabase.table("images").select("*").eq("id", image_id).execute()
    return dict(result.data[0]) if result.data else None


def search_similar(
    query_embedding: list[float],
    top_n: int = 3,
    match_threshold: float = 0.0,
    filters: Optional[dict] = None,
) -> list[dict]:
    """_ + _

    _ Supabase match_images RPC _ pgvector _

    Args:
        query_embedding: _1024 _
        top_n: _
        match_threshold: _0-1_
        filters: _ {"uploader": "...", "date_from": "...", "date_to": "..."}

    Returns:
        list[dict]: _
    """
    import httpx
    base = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    rpc_params = {
        "query_embedding": query_embedding,
        "match_threshold": match_threshold,
        "match_count": top_n,
    }
    if filters:
        if "uploader" in filters:
            rpc_params["filter_uploader"] = filters["uploader"]
        if "date_from" in filters:
            rpc_params["filter_date_from"] = filters["date_from"]
        if "date_to" in filters:
            rpc_params["filter_date_to"] = filters["date_to"]

    import json as _j
    r = httpx.post(
        f"{base}/rest/v1/rpc/match_images",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Accept": "application/json"},
        content=_j.dumps(rpc_params, ensure_ascii=True),
    )
    r.raise_for_status()
    return [dict(item) for item in r.json()]


def count_images() -> int:
    """Return total image count via raw REST (bypasses supabase-py encoding)."""
    import httpx
    base = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    r = httpx.get(f"{base}/rest/v1/images", headers={
        "apikey": key, "Authorization": f"Bearer {key}",
        "Prefer": "count=exact"
    }, params={"select": "id", "limit": 0})
    # PostgREST returns count in Content-Range header
    cr = r.headers.get("content-range", "")
    if cr and "/" in cr:
        return int(cr.rsplit("/", 1)[-1])
    return 0


def clear_all() -> int:
    """_"""
    supabase = get_supabase()
    result = supabase.table("images").delete().neq("id", -1).execute()
    return len(result.data or [])


# ============================================================
# [CN]
# ============================================================

def upload_to_storage(bucket: str, local_path: Path, remote_name: Optional[str] = None) -> str:
    """_ Supabase Storage_ URL_

    _ ASCII _ Storage _ InvalidKey _

    Args:
        bucket: _ ("uploads" | "thumbnails")_
        local_path: _
        remote_name: _

    Returns:
        str: _ URL_
    """
    supabase = get_supabase()

    if remote_name is None:
        remote_name = _safe_storage_name(local_path.name)

    with open(local_path, "rb") as f:
        supabase.storage.from_(bucket).upload(
            path=remote_name,
            file=f,
            file_options={
                "content-type": _guess_mime(local_path),
                "x-upsert": "true",  # _
            },
        )

    public_url = supabase.storage.from_(bucket).get_public_url(remote_name)
    return public_url


def _safe_storage_name(filename: str) -> str:
    """_ Supabase Storage _

    _ ASCII _
    _ ASCII _ hash _
    """
    import hashlib

    stem = Path(filename).stem
    suffix = Path(filename).suffix

    # [CN] ASCII[CN]
    if all(ord(c) < 128 for c in filename):
        return filename

    # [CN] hash [CN]
    name_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    # [CN] ASCII [CN]
    ascii_part = "".join(c for c in stem if c.isascii() and (c.isalnum() or c in "-_."))[:20]
    safe_name = f"{ascii_part}_{name_hash}{suffix}" if ascii_part else f"img_{name_hash}{suffix}"
    return safe_name


def delete_from_storage(bucket: str, remote_name: str) -> None:
    """_ Storage _"""
    supabase = get_supabase()
    supabase.storage.from_(bucket).remove([remote_name])


def _guess_mime(path: Path) -> str:
    """_ MIME _"""
    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return mime_map.get(suffix, "application/octet-stream")


# ============================================================
# [CN]
# ============================================================

def check_connection() -> dict:
    """_ Supabase _"""
    try:
        supabase = get_supabase()
        count = count_images()
        buckets = supabase.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        return {
            "status": "ok",
            "image_count": count,
            "buckets": bucket_names,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================
# [CN]
# ============================================================

if __name__ == "__main__":
    status = check_connection()
    print("Supabase connection test:")
    for k, v in status.items():
        print(f"  {k}: {v}")
