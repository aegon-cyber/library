import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
pdf_processor — PDF 文档解析与摘要生成模块

功能：
  1. 从 PDF 中提取所有嵌入图片
  2. 提取 PDF 全文文本
  3. 调用 AI 生成 300 字文档摘要
  4. 将摘要作为 extra_description 传入图像索引流程

依赖：
  - PyMuPDF (fitz) — PDF 解析
  - zhipuai SDK — 摘要生成
  - upload_test — 图片处理核心

使用示例：
  from pdf_processor import process_pdf
  process_pdf("report.pdf", uploader="万里")
"""

import os
import fitz
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY")
SUMMARY_MODEL = os.getenv("ZHIPU_SUMMARY_MODEL", "GLM-5V-Turbo")

if not API_KEY:
    raise RuntimeError("ZHIPU_API_KEY not found in .env")

client = ZhipuAI(api_key=API_KEY)

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"


def extract_images_from_pdf(pdf_path: str | Path) -> list[Path]:
    """从 PDF 每一页中提取嵌入图片，保存到 uploads/ 目录。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        list[Path]: 提取出的图片文件路径列表。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"文档不存在: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    image_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
            except Exception:
                continue

            dest_name = f"pdf_{timestamp}_p{page_num + 1}_{img_idx + 1}.{ext}"
            dest_path = UPLOADS_DIR / dest_name

            with open(dest_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(dest_path)
            print(f"   Extracted: {dest_name} (page {page_num + 1})")

    doc.close()
    return image_paths


def render_first_page(pdf_path: str | Path) -> Path:
    """将 PDF 第一页渲染为图片（当 PDF 没有嵌入图片时的 fallback）。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        Path: 渲染出的图片路径。
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    # 渲染为 2x 分辨率图片
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_name = f"pdf_{timestamp}_page1.png"
    dest_path = UPLOADS_DIR / dest_name
    pix.save(str(dest_path))
    doc.close()
    print(f"   Rendered first page: {dest_name}")
    return dest_path


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """从 PDF 中提取所有文本内容。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        str: PDF 全文文本。
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    parts = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            parts.append(f"[第{page_num + 1}页]\n{text}")
    doc.close()
    return "\n\n".join(parts)


def generate_pdf_summary(pdf_path: str | Path, max_chars: int = 300) -> str:
    """调用 AI 对 PDF 内容生成摘要。

    Args:
        pdf_path: PDF 文件路径。
        max_chars: 摘要最大字数，默认 300。

    Returns:
        str: 中文摘要。
    """
    pdf_path = Path(pdf_path)
    full_text = extract_text_from_pdf(pdf_path)

    if not full_text.strip():
        return "（文档无文本内容）"

    text_for_summary = full_text[:4000] if len(full_text) > 4000 else full_text

    try:
        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"你是一个文档摘要助手。请根据文档内容生成一段{max_chars}字以内的中文摘要，"
                        f"重点提取核心事实：涉及人物、关键事件、数据指标、结论观点。"
                        f"不要评价，只输出事实。"
                    ),
                },
                {"role": "user", "content": text_for_summary},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"   Warning: AI 摘要生成失败 ({e})，使用原文前{max_chars}字")
        return full_text[:max_chars]


def process_pdf(pdf_path: str, uploader: str = "test_user") -> dict:
    """处理 PDF 导入：提取图片 + 生成摘要 + 向量化入库。

    完整流程：
      1. 上传原始 PDF 到 Supabase Storage（用于来源追溯）
      2. 解析 PDF，提取所有嵌入图片
      3. 如果没有图片，渲染第一页作为封面
      4. 提取全文文本
      5. AI 生成 300 字摘要
      6. 对每张图片调用 process_image()，摘要作为 extra_description，来源指向原始 PDF

    Args:
        pdf_path: PDF 文件路径。
        uploader: 上传人名称。

    Returns:
        dict: {pdf_name, page_count, summary, image_count, image_results}
    """
    from upload_test import process_image
    from supabase_client import upload_to_storage

    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem

    print(f"\n{'=' * 60}")
    print(f"Processing PDF: {pdf_name}")
    print(f"{'=' * 60}")

    # 0. 上传原始 PDF 到 Supabase Storage
    print("\n[Step 0] Uploading original PDF to Storage...")
    try:
        source_url = upload_to_storage("uploads", pdf_path)
        print(f"   Source URL: {source_url}")
    except Exception as e:
        print(f"   Warning: PDF upload failed ({e}), skipping source link")
        source_url = ""

    # 0.5. 获取页数
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    # 1. 提取图片
    print("\n[Step 1] Extracting images...")
    image_paths = extract_images_from_pdf(pdf_path)

    if not image_paths:
        print("   No embedded images found, rendering first page as cover")
        cover_path = render_first_page(pdf_path)
        image_paths = [cover_path]

    print(f"   {len(image_paths)} image(s) to index")

    # 2. 提取文本
    print(f"\n[Step 2] Extracting text ({page_count} pages)...")
    full_text = extract_text_from_pdf(pdf_path)
    text_preview = full_text[:150] + "..." if len(full_text) > 150 else full_text
    print(f"   Text preview: {text_preview}")

    # 3. 生成摘要
    print("\n[Step 3] Generating summary...")
    summary = generate_pdf_summary(pdf_path, max_chars=300)
    print(f"   Summary: {summary}")

    # 4. 每张图片向量化（带来源信息）
    source_label = f"{pdf_name}.pdf"
    print(f"\n[Step 4] Indexing {len(image_paths)} images with summary...")
    image_results = []
    for img_path in image_paths:
        try:
            result = process_image(
                str(img_path),
                uploader=uploader,
                extra_description=f"[PDF摘要|{page_count}页] {summary}",
                source_file=source_label,
                source_url=source_url,
            )
            image_results.append(result)
        except Exception as e:
            print(f"   Failed: {img_path.name} - {e}")

    print(f"\n{'=' * 60}")
    print(f"PDF processing complete: {len(image_results)}/{len(image_paths)} indexed")
    print(f"{'=' * 60}")

    return {
        "pdf_name": pdf_name,
        "page_count": page_count,
        "summary": summary,
        "image_count": len(image_paths),
        "indexed_count": len(image_results),
        "image_results": image_results,
    }


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_file_path> [uploader_name]")
        print("Example: python pdf_processor.py report.pdf 万里")
        sys.exit(1)

    pdf_file = sys.argv[1]
    uploader_name = sys.argv[2] if len(sys.argv) > 2 else "test_user"

    result = process_pdf(pdf_file, uploader=uploader_name)
    print(f"\nFinal: {result['indexed_count']} images indexed")
    print(f"Summary: {result['summary']}")
