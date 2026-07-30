"""
library-agent - 图片向量化检索系统 (Supabase 云端版)

功能：
  1. 上传图片并生成缩略图 → 存入 Supabase Storage
  2. 基于文件名和标签生成向量（智谱AI Embedding） → 存入 Supabase PostgreSQL + pgvector
  3. 语义搜索：pgvector 余弦相似度匹配
  4. 查看所有已上传图片

依赖：
  - zhipuai SDK (智谱AI Embedding API)
  - Pillow (图片处理)
  - python-dotenv (环境变量管理)
  - supabase (数据库 + 存储)

使用前请：
  1. 创建虚拟环境: python -m venv library
  2. 激活虚拟环境: library\\Scripts\\activate.bat (Windows)
  3. 安装依赖: pip install -r requirements.txt && pip install supabase
  4. 创建 .env 文件，填写 ZHIPU_API_KEY + SUPABASE_URL + SUPABASE_KEY
  5. 在 Supabase Dashboard SQL Editor 中执行 setup_supabase.sql
  6. 运行: python upload_test.py
"""

import os
import json
import math
import sys
from datetime import datetime
from pathlib import Path

# 强制 UTF-8，解决 Windows GBK 编码问题
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _safe_print(*args, **kwargs):
    """print 的安全包装：自动处理 GBK 无法编码的字符。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 将无法编码的字符替换为 ?
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
# 环境配置
# ============================================================

# 加载 .env 文件中的环境变量
load_dotenv()

# 读取配置
API_KEY = os.getenv("ZHIPU_API_KEY")
EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")
VISION_MODEL = os.getenv("ZHIPU_VISION_MODEL", "GLM-5V-Turbo")

# 初始化智谱AI客户端
if not API_KEY:
    print("ERROR: ZHIPU_API_KEY not found, please configure in .env file")
    exit(1)

client = ZhipuAI(api_key=API_KEY)

# 项目目录常量
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"       # 本地临时处理目录（上传到 Supabase Storage 后可清理）
THUMBNAILS_DIR = BASE_DIR / "thumbnails" # 本地临时缩略图目录（上传到 Supabase Storage 后可清理）

# 确保本地临时目录存在
UPLOADS_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)


# ============================================================
# 图片处理
# ============================================================

def generate_thumbnail(image_path: str | Path) -> Path:
    """生成图片的缩略图（宽200px，保持比例）。

    Args:
        image_path: 原始图片路径。

    Returns:
        Path: 缩略图保存路径 (thumbnails/thumb_原文件名)。

    Raises:
        FileNotFoundError: 图片文件不存在。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    img = Image.open(image_path)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    original_width, original_height = img.size

    # 计算缩略图尺寸：宽200px，保持比例
    thumb_width = 200
    thumb_height = int(original_height * (thumb_width / original_width))

    img.thumbnail((thumb_width, thumb_height), Image.LANCZOS)

    # 保存缩略图
    thumb_filename = f"thumb_{image_path.name}"
    thumb_path = THUMBNAILS_DIR / thumb_filename
    img.save(thumb_path)

    return thumb_path


def describe_image(image_path: str | Path) -> str:
    """调用 GLM-5V-Turbo 视觉模型识别图片内容，生成文字描述。

    将图片 base64 编码后传给视觉模型，让模型用中文描述图片中的
    主体对象、场景、文字内容、颜色、风格等关键信息。

    Args:
        image_path: 图片文件路径。

    Returns:
        str: 图片的文字描述。

    Raises:
        Exception: API调用失败。
    """
    import base64

    image_path = Path(image_path)

    # 读取图片并 base64 编码
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # 根据文件扩展名确定 MIME 类型
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

    try:
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
        description = response.choices[0].message.content
        return description
    except Exception as e:
        raise Exception(f"GLM-5V-Turbo 图片识别失败: {e}")


def get_embedding(text: str) -> list[float]:
    """调用智谱AI Embedding-3 API对文本生成向量。

    Args:
        text: 文本字符串。

    Returns:
        list[float]: 向量列表。

    Raises:
        Exception: API调用失败。
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Embedding-3 向量化失败: {e}")


def process_image(image_path: str, uploader: str = "test_user", extra_description: str = None,
                  source_file: str = "", source_url: str = "") -> dict:
    """处理图片上传：生成缩略图、多模态向量化、存入 Supabase。

    Args:
        image_path: 图片文件路径。
        uploader: 上传人名称，默认 test_user。
        extra_description: 额外描述文本（人工标注/文档摘要），拼接到视觉描述后一起向量化。
        source_file: 图片来源文档名（如 "report.pdf"），直接上传的图片为空。
        source_url: 图片来源文档的 Supabase Storage 链接。

    Returns:
        dict: 图片信息字典，包含 id, file_name, embedding 等字段。

    Raises:
        FileNotFoundError: 图片文件不存在。
        Exception: 缩略图生成或API调用失败。
    """
    image_path = Path(image_path)

    # 1. 检查文件是否存在
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    print(f"Processing: {image_path.name}")

    # 2. 复制原图到本地临时 uploads/ 目录
    local_dest = UPLOADS_DIR / image_path.name
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    img.save(local_dest)

    # 3. 生成缩略图（本地 Pillow 处理）
    thumb_path = generate_thumbnail(image_path)
    print(f"   Thumbnail: {thumb_path}")

    # 4. 上传原图和缩略图到 Supabase Storage
    print(f"   Uploading to Supabase Storage...")
    try:
        file_url = upload_to_storage("uploads", local_dest)
        thumb_url = upload_to_storage("thumbnails", thumb_path)
        print(f"   File URL: {file_url}")
        print(f"   Thumb URL: {thumb_url}")
    except Exception as e:
        raise Exception(f"Supabase Storage 上传失败: {e}")

    # 5. 先用 GLM-5V-Turbo 识别图片内容，生成文字描述
    description = describe_image(image_path)
    print(f"   Description: {description}")

    # 5.5 如果有额外描述（人工标注/文档摘要），拼接到视觉描述后
    full_description = description
    if extra_description:
        full_description = f"{description}\n{extra_description}"
        print(f"   Extra description: {extra_description}")

    # 6. 再用 Embedding-3 对描述文本生成向量
    embedding = get_embedding(full_description)
    print(f"   Embedding generated, dims: {len(embedding)}")

    # 7. 存入 Supabase PostgreSQL（含向量）
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
# 语义搜索
# ============================================================

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    Args:
        vec_a: 向量A。
        vec_b: 向量B。

    Returns:
        float: 余弦相似度，范围 [-1, 1]。
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def search_similar(query: str, top_n: int = 3, filters: dict = None) -> list[dict]:
    """语义搜索：pgvector 余弦相似度 + 可选结构化过滤。

    Args:
        query: 搜索查询文本。
        top_n: 返回结果数量，默认3。
        filters: 可选的结构化过滤条件，支持:
            - uploader: 上传人精确匹配
            - date_from: 起始日期 (YYYY-MM-DD)
            - date_to: 截止日期 (YYYY-MM-DD)
            示例: {"uploader": "万里", "date_from": "2026-07-01"}

    Returns:
        list[dict]: 按相似度降序排列的结果列表，每项包含
                    file_name, similarity, file_path, id, description。
    """
    print(f"Searching: \"{query}\"")
    if filters:
        print(f"   Filters: {filters}")

    # 1. 将查询文本向量化
    query_embedding = get_embedding(query)
    print(f"   Query embedding generated")

    # 2. 调用 Supabase pgvector RPC 搜索（向量相似度 + 结构化过滤一站式）
    total = db_count_images()
    if total == 0:
        print("   No images in DB, please upload first")
        return []

    results = db_search_similar(query_embedding, top_n=top_n, filters=filters)
    print(f"   pgvector search: {len(results)} results (from {total} images)")

    if not results:
        print("   No images match filters")
        return []

    # 3. 打印结果
    print(f"\nResults (top {len(results)}):")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        upload_time = r.get("upload_time", "")
        if isinstance(upload_time, str):
            upload_time = upload_time[:10]
        print(f"  {i}. {r['file_name']}")
        print(f"     Similarity: {r['similarity']:.4f}")
        print(f"     Path: {r['file_path']}")
        print(f"     Uploader: {r['uploader']} | Time: {upload_time}")
        print()

    return results


def generate_answer(query: str, results: list[dict], top_n: int = 3) -> str:
    """基于检索结果，调用 LLM 生成自然语言回答。

    将检索到的图片描述、文档摘要、上传人、时间等信息拼成上下文，
    让 GLM-5V-Turbo 以自然语言组织回答，直接回应用户的提问。

    Args:
        query: 用户原始提问。
        results: search_similar() 返回的匹配结果列表。
        top_n: 最多使用几条结果生成回答，默认3。

    Returns:
        str: 自然语言回答文本。
    """
    if not results:
        return "知识库中未找到相关内容。"

    # 取前 top_n 条，构建上下文
    context_parts = []
    for i, r in enumerate(results[:top_n], 1):
        part = f"[资料{i}] 来源：{r['uploader']}，时间：{r['upload_time'][:10]}\n"
        part += f"图片描述：{r['description']}\n"
        # 补充 extra_description（从 SQLite 获取）
        extra = r.get("extra_description", "")
        if extra:
            part += f"文档摘要：{extra}\n"
        part += f"相似度：{r['similarity']:.2f}"
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,  # GLM-5V-Turbo 也支持纯文本
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个知识库问答助手。请根据以下检索到的资料回答用户的问题。"
                        "如果资料中包含相关数据（数字、日期、金额、百分比等），请准确引用。"
                        "如果资料不足以回答问题，请如实说明。"
                        "回答要简洁、直接，不超过200字。"
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
        # 降级：直接拼接描述返回
        fallback = "\n".join([f"• {r['file_name']}: {r['description'][:100]}..." for r in results[:top_n]])
        return f"（AI回答生成失败，以下为检索结果摘要）\n{fallback}"


def ask(query: str, top_n: int = 3, filters: dict = None) -> str:
    """RAG 完整流程：向量检索 + LLM 生成自然语言回答。

    先调用 search_similar() 检索匹配图片/文档，
    再调用 generate_answer() 将结果组织成自然语言回答。

    Args:
        query: 用户提问文本。
        top_n: 检索返回数量，默认3。
        filters: 可选结构化过滤条件。

    Returns:
        str: 自然语言回答。
    """
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    if filters:
        print(f"Filters: {filters}")
    print(f"{'='*60}")

    # 1. 向量检索
    results = search_similar(query, top_n=top_n, filters=filters)

    if not results:
        print("\nA: 知识库中未找到相关内容。")
        return "知识库中未找到相关内容。"

    # 2. extra_description 已由 search_similar() 从 Supabase 返回，无需额外查询
    # （match_images RPC 返回的 results 中已包含 extra_description 字段）

    # 3. LLM 生成回答
    print("\nGenerating answer...")
    answer = generate_answer(query, results, top_n=top_n)

    print(f"\nA: {answer}")
    print(f"{'='*60}\n")

    return answer


# ============================================================
# 查看所有图片
# ============================================================

def list_all_images() -> None:
    """显示所有已上传图片的摘要信息。"""
    images = db_get_all_images()

    if not images:
        print("No images in DB")
        return

    print(f"\nTotal {len(images)} images:")
    print("-" * 60)
    for img in images:
        print(f"  ID: {img['id']} | {img['file_name']}")


# ============================================================
# 主交互程序
# ============================================================

def main():
    """主交互菜单程序。"""
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
            # 上传图片
            image_path = input("Image path: ").strip()
            uploader = input("Uploader (default test_user): ").strip() or "test_user"

            try:
                process_image(image_path, uploader)
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "2":
            # 搜索图片
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
            # RAG 问答
            query = input("Your question: ").strip()
            if not query:
                print("Question cannot be empty")
                continue

            top_n_str = input("Top N (default 3): ").strip()
            top_n = int(top_n_str) if top_n_str else 3

            # 可选结构化过滤
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
            # 查看所有图片
            list_all_images()

        elif choice == "5":
            # 退出
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice, enter 1-5")


if __name__ == "__main__":
    main()