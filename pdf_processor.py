import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
pdf_processor — PDF _

_
  1. _ PDF _
  2. _ PDF _
  3. _ AI _ 300 _Doc summary
  4. _ extra_description _

_
  - PyMuPDF (fitz) — PDF _
  - zhipuai SDK — _
  - upload_test — _

_
  from pdf_processor import process_pdf
  process_pdf("report.pdf", uploader="_")
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
    """_ PDF _ uploads/ _

    Args:
        pdf_path: PDF _

    Returns:
        list[Path]: _
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Document not found: {pdf_path}")

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
    """_ PDF _ PDF _ fallback_

    Args:
        pdf_path: PDF _

    Returns:
        Path: _
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    # [CN] 2x [CN]
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
    """_ PDF _

    Args:
        pdf_path: PDF _

    Returns:
        str: PDF _
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    parts = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            parts.append(f"[_{page_num + 1}_]\n{text}")
    doc.close()
    return "\n\n".join(parts)


def generate_pdf_summary(pdf_path: str | Path, max_chars: int = 300) -> str:
    """_ AI _ PDF _

    Args:
        pdf_path: PDF _
        max_chars: _ 300_

    Returns:
        str: _
    """
    pdf_path = Path(pdf_path)
    full_text = extract_text_from_pdf(pdf_path)

    if not full_text.strip():
        return "(no text content)"

    text_for_summary = full_text[:4000] if len(full_text) > 4000 else full_text

    import os as _os, httpx as _httpx
    try:
        ds_key = _os.getenv("DS_API_KEY", "")
        resp = _httpx.post("https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
            json={"model":"deepseek-v4-flash", "messages":[
                {"role":"system","content":f"你是一个文档摘要助手。请根据文档内容生成一段{max_chars}字以内的中文摘要，重点提取核心事实：涉及人物、关键事件、数据指标、结论观点。不要评价，只输出事实。"},
                {"role":"user","content": text_for_summary}
            ], "max_tokens":400},
            timeout=30)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"   Warning: AI summary generation failed ({e})_{max_chars}_")
        return full_text[:max_chars]


def process_pdf(pdf_path: str, uploader: str = "test_user", source_name: str = "") -> dict:
    """_ PDF _ + _ + _

    _
      1. _ PDF _ Supabase Storage_
      2. _ PDF_
      3. _
      4. _
      5. AI _ 300 _
      6. _ process_image()_ extra_description_ PDF

    Args:
        pdf_path: PDF _
        uploader: _

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

    # 0. [CN] PDF [CN] Supabase Storage
    print("\n[Step 0] Uploading original PDF to Storage...")
    try:
        source_url = upload_to_storage("uploads", pdf_path)
        print(f"   Source URL: {source_url}")
    except Exception as e:
        print(f"   Warning: PDF upload failed ({e}), skipping source link")
        source_url = ""

    # 0.5. [CN]
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    # 1. [CN]
    print("\n[Step 1] Extracting images...")
    image_paths = extract_images_from_pdf(pdf_path)

    if not image_paths:
        print("   No embedded images found, rendering first page as cover")
        cover_path = render_first_page(pdf_path)
        image_paths = [cover_path]

    print(f"   {len(image_paths)} image(s) to index")

    # 2. [CN]
    print(f"\n[Step 2] Extracting text ({page_count} pages)...")
    full_text = extract_text_from_pdf(pdf_path)
    text_preview = full_text[:150] + "..." if len(full_text) > 150 else full_text
    print(f"   Text preview: {text_preview}")

    # 3. [CN]
    print("\n[Step 3] Generating summary...")
    summary = generate_pdf_summary(pdf_path, max_chars=300)
    print(f"   Summary: {summary}")

    # 4. [CN]
    source_label = source_name if source_name else f"{pdf_name}.pdf"
    print(f"\n[Step 4] Indexing {len(image_paths)} images with summary...")
    image_results = []
    for img_path in image_paths:
        try:
            result = process_image(
                str(img_path),
                uploader=uploader,
                extra_description=f"[PDF summary|{page_count}_] {summary}",
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
# [CN]
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_file_path> [uploader_name]")
        print("Example: python pdf_processor.py report.pdf _")
        sys.exit(1)

    pdf_file = sys.argv[1]
    uploader_name = sys.argv[2] if len(sys.argv) > 2 else "test_user"

    result = process_pdf(pdf_file, uploader=uploader_name)
    print(f"\nFinal: {result['indexed_count']} images indexed")
    print(f"Summary: {result['summary']}")
