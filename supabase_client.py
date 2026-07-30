"""
supabase_client - Supabase 数据库 + 存储客户端

替代本地的 data.json + images.db + uploads/thumbnails/ 目录。
所有操作通过 Supabase REST API 完成（不直连 PostgreSQL）。

表：images (含 pgvector embedding 列)
存储桶：uploads / thumbnails
搜索函数：match_images (RPC)
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    """获取 Supabase 客户端单例。"""
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


# ============================================================
# 数据库操作
# ============================================================

def add_image(image_info: dict) -> dict:
    """向 Supabase 写入一条图片记录（含向量）。

    Args:
        image_info: 包含 file_name, file_path, thumbnail_path, uploader,
                    upload_time, description, extra_description, embedding

    Returns:
        dict: 写入后的完整记录（含服务端生成的 id）
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
    """获取所有图片记录（不含 embedding，减少传输）。"""
    supabase = get_supabase()
    result = supabase.table("images").select(
        "id, file_name, file_path, thumbnail_path, uploader, upload_time, description, extra_description, source_file, source_url, duplicate_of"
    ).order("id", desc=False).execute()
    return [dict(r) for r in (result.data or [])]


def get_image_by_id(image_id: int) -> Optional[dict]:
    """根据 ID 获取单条图片记录。"""
    supabase = get_supabase()
    result = supabase.table("images").select("*").eq("id", image_id).execute()
    return dict(result.data[0]) if result.data else None


def search_similar(
    query_embedding: list[float],
    top_n: int = 3,
    match_threshold: float = 0.0,
    filters: Optional[dict] = None,
) -> list[dict]:
    """向量相似度搜索 + 可选结构化过滤。

    调用 Supabase match_images RPC 函数，在数据库内完成 pgvector 余弦距离计算。

    Args:
        query_embedding: 查询向量（1024 维）。
        top_n: 返回数量。
        match_threshold: 最低相似度阈值（0-1）。
        filters: 可选 {"uploader": "...", "date_from": "...", "date_to": "..."}

    Returns:
        list[dict]: 按相似度降序的结果列表。
    """
    supabase = get_supabase()

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

    result = supabase.rpc("match_images", rpc_params).execute()
    return [dict(r) for r in (result.data or [])]


def count_images() -> int:
    """返回图片总数。"""
    supabase = get_supabase()
    result = supabase.table("images").select("id", count="exact").execute()
    return result.count or 0


def clear_all() -> int:
    """清空所有图片记录（调试用）。返回删除条数。"""
    supabase = get_supabase()
    result = supabase.table("images").delete().neq("id", -1).execute()
    return len(result.data or [])


# ============================================================
# 存储操作
# ============================================================

def upload_to_storage(bucket: str, local_path: Path, remote_name: Optional[str] = None) -> str:
    """上传文件到 Supabase Storage，返回公开 URL。

    文件名中的非 ASCII 字符会被转换为安全形式，避免 Storage 的 InvalidKey 错误。

    Args:
        bucket: 存储桶名 ("uploads" | "thumbnails")。
        local_path: 本地文件路径。
        remote_name: 远程文件名，默认使用本地文件名（自动净化）。

    Returns:
        str: 文件公开访问 URL。
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
                "x-upsert": "true",  # 覆盖同名文件
            },
        )

    public_url = supabase.storage.from_(bucket).get_public_url(remote_name)
    return public_url


def _safe_storage_name(filename: str) -> str:
    """将文件名转换为 Supabase Storage 安全格式。

    保留扩展名，主体部分只保留 ASCII 字母数字、连字符、下划线。
    非 ASCII 字符用文件名的 hash 替代，确保唯一且安全。
    """
    import hashlib

    stem = Path(filename).stem
    suffix = Path(filename).suffix

    # 如果文件名已经是纯 ASCII，直接返回
    if all(ord(c) < 128 for c in filename):
        return filename

    # 否则用 hash 生成安全文件名
    name_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    # 尽量保留原始文件名中的 ASCII 部分
    ascii_part = "".join(c for c in stem if c.isascii() and (c.isalnum() or c in "-_."))[:20]
    safe_name = f"{ascii_part}_{name_hash}{suffix}" if ascii_part else f"img_{name_hash}{suffix}"
    return safe_name


def delete_from_storage(bucket: str, remote_name: str) -> None:
    """从 Storage 删除文件。"""
    supabase = get_supabase()
    supabase.storage.from_(bucket).remove([remote_name])


def _guess_mime(path: Path) -> str:
    """根据扩展名推断 MIME 类型。"""
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
# 快速诊断
# ============================================================

def check_connection() -> dict:
    """测试 Supabase 连接状态。"""
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
# 命令行测试入口
# ============================================================

if __name__ == "__main__":
    status = check_connection()
    print("Supabase connection test:")
    for k, v in status.items():
        print(f"  {k}: {v}")
