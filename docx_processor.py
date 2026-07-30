import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
docx_processor - Word 文档解析与摘要生成模块

功能：
  1. 从 Word 文档中提取所有嵌入图片
  2. 提取文档全文文本
  3. 调用 AI 生成 300 字文档摘要（涵盖主题、人物、关键数据）
  4. 将摘要作为 extra_description 传入图像索引流程

依赖：
  - python-docx (Word 文档解析)
  - zhipuai SDK (摘要生成)
  - upload_test (图片处理核心)

使用示例：
  from docx_processor import process_docx
  process_docx("report.docx", uploader="万里")
"""

import os
import base64
from pathlib import Path
from datetime import datetime
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY")
SUMMARY_MODEL = os.getenv("ZHIPU_SUMMARY_MODEL", "GLM-5V-Turbo")

if not API_KEY:
    raise RuntimeError("ZHIPU_API_KEY not found in .env")

client = ZhipuAI(api_key=API_KEY)

# 项目目录常量
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp_docx"


def extract_images_from_docx(docx_path: str | Path) -> list[Path]:
    """从 Word 文档中提取所有嵌入图片，保存到 uploads/ 目录。

    遍历文档中所有图片关系（IMAGE 类型），将图片二进制数据
    导出为独立文件。

    Args:
        docx_path: Word 文档路径 (.docx)。

    Returns:
        list[Path]: 提取出的图片文件路径列表。
    """
    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"文档不存在: {docx_path}")

    doc = Document(docx_path)
    image_paths = []

    # 遍历所有图片关系
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            image_data = rel.target_part.blob

            # 尝试从文件名推断扩展名
            target_name = Path(rel.target_part.partname).name
            suffix = Path(target_name).suffix or ".png"

            # 保存到 uploads/，用时间戳避免重名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_name = f"docx_{timestamp}_{len(image_paths)+1}{suffix}"
            dest_path = UPLOADS_DIR / dest_name

            with open(dest_path, "wb") as f:
                f.write(image_data)

            image_paths.append(dest_path)
            print(f"   Extracted: {dest_name}")

    return image_paths


def extract_text_from_docx(docx_path: str | Path) -> str:
    """从 Word 文档中提取所有文本内容。

    遍历文档中所有段落，拼接为完整文本。

    Args:
        docx_path: Word 文档路径 (.docx)。

    Returns:
        str: 文档全文文本。
    """
    docx_path = Path(docx_path)
    doc = Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def generate_docx_summary(docx_path: str | Path, max_chars: int = 300) -> str:
    """调用 AI 模型对 Word 文档内容生成摘要。

    读取文档全文文本，让 AI 提炼出与图片相关的核心事实，
    包括：主题、涉及人物、关键事件、数据指标等。

    Args:
        docx_path: Word 文档路径 (.docx)。
        max_chars: 摘要最大字数，默认 300。

    Returns:
        str: 文档摘要文本（中文）。
    """
    docx_path = Path(docx_path)
    full_text = extract_text_from_docx(docx_path)

    if not full_text.strip():
        return "（文档无文本内容）"

    # 如果文本太长，截断到合理范围（避免超 token）
    text_for_summary = full_text[:3000] if len(full_text) > 3000 else full_text

    try:
        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"你是一个新闻文档摘要助手。请根据文档内容生成一段{max_chars}字以内的中文摘要，"
                           f"重点提取与图片直接相关的核心事实：涉及人物、关键事件、数据指标、新闻角度。"
                           f"不要评价，只输出事实。"
                },
                {
                    "role": "user",
                    "content": text_for_summary
                }
            ]
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"   Warning: AI 摘要生成失败 ({e})，使用原文前300字作为摘要")
        return full_text[:max_chars]


def process_docx(docx_path: str, uploader: str = "test_user") -> dict:
    """处理 Word 文档导入：提取图片 + 生成摘要 + 向量化入库。

    完整流程：
      1. 解析 Word，提取所有嵌入图片
      2. 提取文档全文文本
      3. AI 生成 300 字文档摘要
      4. 对每张图片调用 process_image()，将摘要作为 extra_description
         拼接到视觉描述后一起向量化

    Args:
        docx_path: Word 文档路径 (.docx)。
        uploader: 上传人名称。

    Returns:
        dict: 包含 summary, image_count, image_results 的处理结果。
    """
    from upload_test import process_image
    from supabase_client import upload_to_storage

    docx_path = Path(docx_path)
    docx_name = docx_path.stem

    print(f"\n{'='*60}")
    print(f"Processing DOCX: {docx_name}")
    print(f"{'='*60}")

    # 0. 上传原始 DOCX 到 Supabase Storage
    print("\n[Step 0] Uploading original DOCX to Storage...")
    try:
        source_url = upload_to_storage("uploads", docx_path)
        print(f"   Source URL: {source_url}")
    except Exception as e:
        print(f"   Warning: DOCX upload failed ({e}), skipping source link")
        source_url = ""

    # 1. 提取图片
    print("\n[Step 1] Extracting images...")
    image_paths = extract_images_from_docx(docx_path)
    print(f"   Found {len(image_paths)} images")

    # 2. 提取文本
    print("\n[Step 2] Extracting text...")
    full_text = extract_text_from_docx(docx_path)
    text_preview = full_text[:100] + "..." if len(full_text) > 100 else full_text
    print(f"   Text preview: {text_preview}")

    # 3. 生成文档摘要
    print("\n[Step 3] Generating summary...")
    summary = generate_docx_summary(docx_path, max_chars=300)
    print(f"   Summary: {summary}")

    # 4. 对每张图片向量化，传入文档摘要 + 来源
    source_label = f"{docx_name}.docx"
    print(f"\n[Step 4] Indexing {len(image_paths)} images with summary...")
    image_results = []
    for img_path in image_paths:
        try:
            result = process_image(
                str(img_path),
                uploader=uploader,
                extra_description=f"[文档摘要] {summary}",
                source_file=source_label,
                source_url=source_url,
            )
            image_results.append(result)
        except Exception as e:
            print(f"   Failed: {img_path.name} - {e}")

    print(f"\n{'='*60}")
    print(f"DOCX processing complete: {len(image_results)}/{len(image_paths)} images indexed")
    print(f"{'='*60}")

    return {
        "docx_name": docx_name,
        "summary": summary,
        "image_count": len(image_paths),
        "image_results": image_results
    }


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python docx_processor.py <docx_file_path> [uploader_name]")
        print("Example: python docx_processor.py report.docx 万里")
        sys.exit(1)

    docx_file = sys.argv[1]
    uploader_name = sys.argv[2] if len(sys.argv) > 2 else "test_user"

    result = process_docx(docx_file, uploader=uploader_name)
    print(f"\nFinal result: {result['image_count']} images indexed")
    print(f"Summary: {result['summary']}")