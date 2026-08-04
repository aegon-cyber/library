"""
library-agent - _ (Supabase _)

_
  1. _ → _ Supabase Storage
  2. _AI Embedding_ → _ Supabase PostgreSQL + pgvector
  3. _pgvector _
  4. _

_
  - zhipuai SDK (_AI Embedding API)
  - Pillow (_)
  - python-dotenv (_)
  - supabase (_ + _)

使用前请：
  1. _: python -m venv library
  2. _: library\\Scripts\\activate.bat (Windows)
  3. _: pip install -r requirements.txt && pip install supabase
  4. _ .env _ ZHIPU_API_KEY + SUPABASE_URL + SUPABASE_KEY
  5. _ Supabase Dashboard SQL Editor _ setup_supabase.sql
  6. _: python upload_test.py
"""

import os
import json
import math
import sys
from datetime import datetime
from pathlib import Path

# [CN] UTF-8[CN] Windows GBK [CN]
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _safe_print(*args, **kwargs):
    """print _ GBK _"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # [CN] ?
        safe_args = [str(a).encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8') for a in args]
        print(*safe_args, **kwargs)

from dotenv import load_dotenv
from PIL import Image
from zhipuai import ZhipuAI
from supabase_client import (
    get_supabase,
    upload_to_storage,
    add_image as db_add_image,
    search_similar as db_search_similar,
    get_all_images as db_get_all_images,
    count_images as db_count_images,
)

# ============================================================
# [CN]
# ============================================================

# [CN] .env [CN]
load_dotenv()

# [CN]
API_KEY = os.getenv("ZHIPU_API_KEY")
EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")
VISION_MODEL = os.getenv("ZHIPU_VISION_MODEL", "GLM-5V-Turbo")

# [CN]AI[CN]
if not API_KEY:
    print("ERROR: ZHIPU_API_KEY not found, please configure in .env file")
    exit(1)

client = ZhipuAI(api_key=API_KEY)

# [CN]
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"       # _ Supabase Storage _
THUMBNAILS_DIR = BASE_DIR / "thumbnails" # _ Supabase Storage _

# [CN]
UPLOADS_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)


# ============================================================
# [CN]
# ============================================================

def generate_thumbnail(image_path: str | Path) -> Path:
    """_200px_

    Args:
        image_path: _

    Returns:
        Path: _ (thumbnails/thumb__)_

    Raises:
        FileNotFoundError: Image file not found。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    original_width, original_height = img.size

    # [CN]200px[CN]
    thumb_width = 200
    thumb_height = int(original_height * (thumb_width / original_width))

    img.thumbnail((thumb_width, thumb_height), Image.LANCZOS)

    # [CN]
    thumb_filename = f"thumb_{image_path.name}"
    thumb_path = THUMBNAILS_DIR / thumb_filename
    img.save(thumb_path)

    return thumb_path


def describe_image(image_path: str | Path) -> str:
    """_ GLM-5V-Turbo _

    _ base64 _
    _

    Args:
        image_path: _

    Returns:
        str: _

    Raises:
        Exception: API_
    """
    import base64

    image_path = Path(image_path)

    # [CN] base64 [CN]
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # [CN] MIME [CN]
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp"
    }
    mime_type = mime_map.get(suffix, "image/png")

    def _call():
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个图片内容识别助手。请用简洁的中文描述图片中的主体对象、场景、文字内容、颜色、风格等关键信息，输出一段100字以内的描述。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    try:
        return _retry(_call)
    except Exception as e:
        raise Exception(f"GLM-5V-Turbo vision recognition failed: {e}")


import time as _time

def _retry(func, *args, max_retries=3, **kwargs):
    """Call func with retry on rate limit (429)."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if ('429' in msg or '1302' in msg or '速率限制' in msg) and attempt < max_retries - 1:
                _time.sleep(3)
                continue
            raise


def get_embedding(text: str) -> list[float]:
    """_AI Embedding-3 API_"""
    def _call():
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding
    try:
        return _retry(_call)
    except Exception as e:
        raise Exception(f"Embedding-3 embedding failed: {e}")


def process_image(image_path: str, uploader: str = "test_user", extra_description: str = None,
                  source_file: str = "", source_url: str = "") -> dict:
    """_ Supabase_

    Args:
        image_path: _
        uploader: _ test_user_
        extra_description: _/Doc summary_
        source_file: _ "report.pdf"_
        source_url: _ Supabase Storage _

    Returns:
        dict: _ id, file_name, embedding _

    Raises:
        FileNotFoundError: Image file not found。
        Exception: _API_
    """
    image_path = Path(image_path)

    # 1. [CN]
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"Processing: {image_path.name}")

    # 2. [CN] uploads/ [CN]
    local_dest = UPLOADS_DIR / image_path.name
    # Only copy if source != destination (avoids self-copy lock)
    import shutil
    if image_path.resolve() != local_dest.resolve():
        shutil.copy2(image_path, local_dest)
    img = Image.open(local_dest)

    # 3. [CN] Pillow [CN]
    thumb_path = generate_thumbnail(image_path)
    print(f"   Thumbnail: {thumb_path}")

    # 4. [CN] Supabase Storage
    print(f"   Uploading to Supabase Storage...")
    try:
        file_url = upload_to_storage("uploads", local_dest)
        thumb_url = upload_to_storage("thumbnails", thumb_path)
        print(f"   File URL: {file_url}")
        print(f"   Thumb URL: {thumb_url}")
    except Exception as e:
        raise Exception(f"Supabase Storage upload failed: {e}")

    # 5. [CN] GLM-5V-Turbo [CN]
    description = describe_image(image_path)
    print(f"   Description: {description}")

    # 5.5 [CN]/Doc summary[CN]
    full_description = description
    if extra_description:
        full_description = f"{description}\n{extra_description}"
        print(f"   Extra description: {extra_description}")

    # 6. [CN] Embedding-3 [CN]
    embedding = get_embedding(full_description)
    print(f"   Embedding generated, dims: {len(embedding)}")

    # 7. [CN] Supabase PostgreSQL[CN]
    upload_time = datetime.now().isoformat()

    image_info = {
        "file_name": image_path.name,
        "file_path": file_url,
        "thumbnail_path": thumb_url,
        "uploader": uploader,
        "upload_time": upload_time,
        "description": description,
        "extra_description": extra_description,
        "source_file": source_file,
        "source_url": source_url,
        "embedding": embedding,
    }

    result = db_add_image(image_info)
    image_id = result.get("id")
    image_info["id"] = image_id

    print(f"   Saved to Supabase, ID: {image_id}")
    print(f"Done!")

    return image_info


# ============================================================
# [CN]
# ============================================================

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """_

    Args:
        vec_a: _A_
        vec_b: _B_

    Returns:
        float: _ [-1, 1]_
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def search_similar(query: str, top_n: int = 3, filters: dict = None) -> list[dict]:
    """_pgvector _ + _

    Args:
        query: _
        top_n: _3_
        filters: _:
            - uploader: _
            - date_from: _ (YYYY-MM-DD)
            - date_to: _ (YYYY-MM-DD)
            _: {"uploader": "_", "date_from": "2026-07-01"}

    Returns:
        list[dict]: _
                    file_name, similarity, file_path, id, description。
    """
    print(f"Searching: \"{query}\"")
    if filters:
        print(f"   Filters: {filters}")

    # 1. [CN]
    query_embedding = get_embedding(query)
    print(f"   Query embedding generated")

    # 2. [CN] Supabase pgvector RPC [CN] + [CN]
    total = db_count_images()
    if total == 0:
        print("   No images in DB, please upload first")
        return []

    results = db_search_similar(query_embedding, top_n=top_n, filters=filters)
    print(f"   pgvector search: {len(results)} results (from {total} images)")

    if not results:
        print("   No images match filters")
        return []

    # Source document boosting: pull in same-source images
    from supabase_client import get_all_images as _ga
    seen_ids = {r["id"] for r in results}
    all_imgs = _ga()
    source_map = {}
    for img in all_imgs:
        sf = img.get("source_file", "") or ""
        if sf not in source_map:
            source_map[sf] = []
        source_map[sf].append(img)
    boosted = []
    for r in results:
        boosted.append(r)
        sf = r.get("source_file", "")
        if sf and sf in source_map and len(boosted) < top_n * 3:
            for sib in source_map[sf]:
                if sib["id"] not in seen_ids and not sib.get("duplicate_of"):
                    sib["similarity"] = round(r["similarity"] * 0.95, 4)
                    sib["boosted"] = True
                    boosted.append(sib)
                    seen_ids.add(sib["id"])
    results = boosted[: max(top_n * 3, 20)]

    # 3. Print results
    print(f"\nResults (top {len(results)}, source-boosted):")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        upload_time = r.get("upload_time", "")
        if isinstance(upload_time, str):
            upload_time = upload_time[:10]
        boost_tag = " [same doc]" if r.get("boosted") else ""
        print(f"  {i}. {r['file_name']}{boost_tag}")
        print(f"     Similarity: {r['similarity']:.4f}")
        print(f"     Path: {r['file_path']}")
        print(f"     Uploader: {r['uploader']} | Time: {upload_time}")
        print()

    return results


def generate_answer(query: str, results: list[dict], top_n: int = 3) -> str:
    """_ LLM _

    _Doc summary_
    _ GLM-5V-Turbo _

    Args:
        query: _
        results: search_similar() _
        top_n: _3_

    Returns:
        str: _
    """
    if not results:
        return "No relevant content found."

    # [CN] top_n [CN]
    context_parts = []
    for i, r in enumerate(results[:top_n], 1):
        part = f"[_{i}] _{r['uploader']}_{r['upload_time'][:10]}\n"
        part += f"_{r['description']}\n"
        # [CN] extra_description[CN] SQLite [CN]
        extra = r.get("extra_description", "")
        if extra:
            part += f"Doc summary：{extra}\n"
        part += f"_{r['similarity']:.2f}"
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,  # GLM-5V-Turbo _
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个知识库问答助手。请根据以下检索到的资料回答用户的问题。"
                        "如果资料中包含相关数据（数字、日期、金额、百分比等），请准确引用。"
                        "如果资料不足以回答问题，请如实说明。"
                        "_200_"
                    )
                },
                {
                    "role": "user",
                    "content": f"用户提问：{query}\n\n参考资料：\n{context}"
                }
            ]
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        # [CN]
        fallback = "\n".join([f"• {r['file_name']}: {r['description'][:100]}..." for r in results[:top_n]])
        return f"_AI_\n{fallback}"


def ask(query: str, top_n: int = 3, filters: dict = None) -> str:
    """RAG _ + LLM _

    _ search_similar() _/_
    _ generate_answer() _

    Args:
        query: _
        top_n: _3_
        filters: _

    Returns:
        str: _
    """
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    if filters:
        print(f"Filters: {filters}")
    print(f"{'='*60}")

    # 1. [CN]
    results = search_similar(query, top_n=top_n, filters=filters)

    if not results:
        print("\nA: No relevant content found.")
        return "No relevant content found."

    # 2. extra_description [CN] search_similar() [CN] Supabase [CN]
    # [CN]match_images RPC [CN] results [CN] extra_description [CN]

    # 3. LLM [CN]
    print("\nGenerating answer...")
    answer = generate_answer(query, results, top_n=top_n)

    print(f"\nA: {answer}")
    print(f"{'='*60}\n")

    return answer


# ============================================================
# [CN]
# ============================================================

def list_all_images() -> None:
    """_"""
    images = db_get_all_images()

    if not images:
        print("No images in DB")
        return

    print(f"\nTotal {len(images)} images:")
    print("-" * 60)
    for img in images:
        print(f"  ID: {img['id']} | {img['file_name']}")


# ============================================================
# [CN]
# ============================================================

def main():
    """_"""
    print("=" * 50)
    print("Library Agent - Image Vector Search")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("  1. Upload an image")
        print("  2. Search images")
        print("  3. Ask a question (RAG)")
        print("  4. List all images")
        print("  5. Exit")

        choice = input("\nEnter number: ").strip()

        if choice == "1":
            # [CN]
            image_path = input("Image path: ").strip()
            uploader = input("Uploader (default test_user): ").strip() or "test_user"

            try:
                process_image(image_path, uploader)
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "2":
            # [CN]
            query = input("Search query: ").strip()
            if not query:
                print("Query cannot be empty")
                continue

            top_n_str = input("Top N (default 3): ").strip()
            top_n = int(top_n_str) if top_n_str else 3

            try:
                search_similar(query, top_n)
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "3":
            # RAG [CN]
            query = input("Your question: ").strip()
            if not query:
                print("Question cannot be empty")
                continue

            top_n_str = input("Top N (default 3): ").strip()
            top_n = int(top_n_str) if top_n_str else 3

            # [CN]
            use_filter = input("Filter by uploader? (press Enter to skip): ").strip()
            filters = None
            if use_filter:
                filters = {"uploader": use_filter}
                date_from = input("Date from (YYYY-MM-DD, Enter to skip): ").strip()
                if date_from:
                    filters["date_from"] = date_from
                date_to = input("Date to (YYYY-MM-DD, Enter to skip): ").strip()
                if date_to:
                    filters["date_to"] = date_to

            try:
                ask(query, top_n, filters=filters)
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "4":
            # [CN]
            list_all_images()

        elif choice == "5":
            # [CN]
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice, enter 1-5")


if __name__ == "__main__":
    main()