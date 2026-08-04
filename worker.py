"""
worker.py — 独立文件处理进程

扫描 OSS incoming/ 目录，发现新文件后处理入库。
和 Flask server.py 互不依赖，独立运行。

启动: python worker.py
"""

import sys, os, time, json, hashlib, tempfile
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

from supabase_client import _get_oss, OSS_BUCKET
from upload_test import process_image
from docx_processor import process_docx
from pdf_processor import process_pdf

POLL_INTERVAL = 15  # seconds between scans
PREFIX = "incoming/"


def list_pending():
    """List pending files in OSS incoming folder."""
    s3 = _get_oss()
    try:
        resp = s3.list_objects_v2(Bucket=OSS_BUCKET, Prefix=PREFIX, MaxKeys=50)
        files = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".meta.json"):
                continue  # skip metadata sidecars
            files.append(key)
        return files
    except Exception:
        return []


def download(key):
    """Download a file from OSS to temp."""
    s3 = _get_oss()
    suffix = Path(key).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    s3.download_fileobj(OSS_BUCKET, key, tmp)
    tmp.close()
    return Path(tmp.name)


def get_meta(key):
    """Get the metadata sidecar for a file."""
    s3 = _get_oss()
    meta_key = key + ".meta.json"
    try:
        resp = s3.get_object(Bucket=OSS_BUCKET, Key=meta_key)
        return json.loads(resp["Body"].read())
    except Exception:
        return {"original_name": Path(key).name, "uploader": "worker"}


def mark_processed(key):
    """Move file from incoming to processed."""
    s3 = _get_oss()
    dest = key.replace("incoming/", "processed/")
    # Copy then delete
    try:
        s3.copy_object(Bucket=OSS_BUCKET, CopySource={"Bucket": OSS_BUCKET, "Key": key}, Key=dest)
        s3.delete_object(Bucket=OSS_BUCKET, Key=key)
        s3.delete_object(Bucket=OSS_BUCKET, Key=key + ".meta.json")
        return True
    except Exception as e:
        print(f"  Move failed: {e}")
        return False


def process_one(key):
    """Process a single file."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing: {key}")
    meta = get_meta(key)
    fn = meta.get("original_name", Path(key).name)
    uploader = meta.get("uploader", "worker")
    suffix = Path(fn).suffix.lower()
    local = download(key)

    try:
        if suffix == ".docx":
            r = process_docx(str(local), uploader=uploader, source_name=fn)
            print(f"  DOCX: {len(r.get('image_results', []))}/{r.get('image_count', 0)} indexed")
        elif suffix == ".pdf":
            r = process_pdf(str(local), uploader=uploader, source_name=fn)
            print(f"  PDF: {r.get('indexed_count', 0)} indexed")
        elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            r = process_image(str(local), uploader=uploader, source_file=fn)
            print(f"  Image: id={r.get('id')}")
        else:
            print(f"  Skipped: unsupported format {suffix}")
            return
        mark_processed(key)
        print(f"  Done.")
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
    finally:
        try: local.unlink()
        except: pass


if __name__ == "__main__":
    print("=" * 50)
    print("Worker started, scanning for files...")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print(f"  OSS Bucket: {OSS_BUCKET}")
    print("=" * 50)

    while True:
        try:
            pending = list_pending()
            for key in pending:
                try:
                    process_one(key)
                except Exception as e:
                    print(f"  Process error: {e}")
        except Exception as e:
            print(f"Scan error: {e}")
        time.sleep(POLL_INTERVAL)
