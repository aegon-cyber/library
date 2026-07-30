"""
server.py — library-agent API _

_ REST API _ ZhipuAI + Supabase _

_: python server.py
_: 5050
"""

import io
import sys
import os as _os

# [CN] UTF-8[CN] Windows GBK [CN]
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
_os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# [CN] print[CN] print [CN] UTF-8-safe [CN]
import builtins
_real_print = builtins.print
def _utf8_print(*args, **kwargs):
    try:
        _real_print(*args, **kwargs)
    except (UnicodeEncodeError, UnicodeDecodeError):
        safe = []
        for a in args:
            s = str(a)
            try:
                s.encode(sys.stdout.encoding or 'utf-8')
            except UnicodeEncodeError:
                s = s.encode('utf-8', errors='replace').decode('utf-8')
            safe.append(s)
        _real_print(*safe, **kwargs)
builtins.print = _utf8_print
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from upload_test import get_embedding, describe_image, search_similar, client, VISION_MODEL
from supabase_client import (
    upload_to_storage,
    add_image as db_add_image,
    get_all_images as db_get_all_images,
)
from datetime import datetime
import json as _json

import traceback as _tb

def _err(e):
    """Return a JSON 500 with traceback."""
    tb = _tb.format_exc().split('\n')
    return jsonify({"error": str(e), "trace": [l for l in tb[-12:] if l.strip()]}), 500

app = Flask(__name__, static_folder=None)

# Force ALL JSON output to ASCII-safe (Flask 3.x compat)
from flask.json.provider import DefaultJSONProvider
import json as _stdlib_json
class _ASCIIProvider(DefaultJSONProvider):
    @staticmethod
    def dumps(obj, **kw):
        kw.setdefault('ensure_ascii', True)
        return _stdlib_json.dumps(obj, **kw)
app.json = _ASCIIProvider(app)

CORS(app)

# ── Global traceback reporter ──
@app.errorhandler(500)
@app.errorhandler(Exception)
def handle_all_errors(e):
    tb_lines = _tb.format_exc().split('\\n')
    return jsonify({
        "error": str(e),
        "traceback": [l for l in tb_lines[-15:] if l.strip()],
    }), 500


# [CN]
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

THUMB_DIR = BASE_DIR / "thumbnails"
THUMB_DIR.mkdir(exist_ok=True)


# ============================================================
# [CN]
# ============================================================

def parse_query(raw_input: str) -> dict:
    """_ LLM _

    _ "_"
      → {query: "_", uploader: "_", date_from: "2026-07-01", date_to: "2026-07-31"}

    _ None_
    """
    # [CN] LLM [CN]
    try:
        all_images = db_get_all_images()
        known_uploaders = list(set(img.get("uploader", "") for img in all_images if img.get("uploader")))
    except Exception:
        known_uploaders = []

    uploader_hint = ""
    if known_uploaders:
        uploader_hint = f"\n_{_json.dumps(known_uploaders, ensure_ascii=False)}"

    prompt = f"""你是一个搜索意图解析器。从用户输入中提取以下信息，输出纯JSON：

1. query: _
2. uploader: _null
3. date_from / date_to: _ YYYY-MM-DD _null
   - "_" → _7_1_~7_31_
   - "_" / "_" → _ {datetime.now().strftime('%Y-%m-%d')} _
   - "_" → 30_

⚠️ _ uploader_"_""_"_"_""_"_null_
{uploader_hint}

_{raw_input}

_JSON_"""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,  # _
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        # [CN] markdown [CN]
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
        parsed = _json.loads(text)
        return {
            "query": parsed.get("query") or raw_input,
            "uploader": parsed.get("uploader"),
            "date_from": parsed.get("date_from"),
            "date_to": parsed.get("date_to"),
        }
    except Exception:
        # [CN]
        return {"query": raw_input, "uploader": None, "date_from": None, "date_to": None}


# ============================================================
# API [CN]
# ============================================================

@app.route("/api/process", methods=["POST"])
def api_process():
    """_ → AI_ → _ → Supabase_PDF_DOCX_"""
    try:
        return _handle_process()
    except UnicodeEncodeError:
        return "GBK encoding conflict on Windows", 500
    except Exception as e:
        msg = str(e).encode('ascii', errors='replace').decode('ascii')
        return jsonify({"error": msg}), 500


import logging as _logging
_logging.getLogger('werkzeug').setLevel(_logging.ERROR)

def _handle_process():
    # ── [CN] ──
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file = request.files["file"]
    uploader = request.form.get("uploader", "test_user")

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Sanitize filename for safe temp storage
    fn = file.filename.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    try:
        safe_fn = fn.encode('ascii', errors='replace').decode('ascii').replace('?', '')
    except (UnicodeEncodeError, UnicodeDecodeError):
        safe_fn = fn.encode('utf-8', errors='replace').decode('ascii', errors='replace').replace('?', '')
    if not safe_fn or not safe_fn.strip('._-'):
        import hashlib
        safe_fn = 'upload_' + hashlib.md5(fn.encode('utf-8')).hexdigest()[:8]
    # [CN]
    if '.' in fn:
        _, ext = fn.rsplit('.', 1)
        if ext.lower() in ('docx', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'):
            safe_fn = safe_fn.rsplit('.', 1)[0] + '.' + ext.lower()

    suffix = '.' + fn.rsplit('.', 1)[-1].lower() if '.' in fn else ''
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    file.save(tmp.name)
    tmp.close()
    temp_path = Path(tmp.name)

    extra_description = None

    # ── PDF [CN] ──
    if suffix == ".pdf":
        try:
            from pdf_processor import process_pdf
            result = process_pdf(temp_path.as_posix(), uploader=uploader)
            return jsonify({
                "type": "pdf",
                "name": result.get("pdf_name", ""),
                "page_count": result.get("page_count", 0),
                "summary": result.get("summary", ""),
                "image_count": result.get("image_count", 0),
                "indexed_count": result.get("indexed_count", 0),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return _err(e)

    # ── DOCX [CN] ──
    if suffix == ".docx":
        try:
            from docx_processor import process_docx
            result = process_docx(temp_path.as_posix(), uploader=uploader)
            return jsonify({
                "type": "docx",
                "name": result.get("docx_name", ""),
                "summary": str(result.get("summary", "")),
                "image_count": result.get("image_count", 0),
                "indexed_count": len(result.get("image_results", [])),
            })
        except Exception as e:
            return _err(e)

    # ── [CN] ──
    img_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    if suffix not in img_exts:
        return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

    try:
        # 1. [CN]
        img = Image.open(temp_path)
        thumb_width = 200
        thumb_height = int(img.height * (thumb_width / img.width))
        img.thumbnail((thumb_width, thumb_height), Image.LANCZOS)
        thumb_path = THUMB_DIR / f"thumb_{temp_path.stem}.png"
        img.save(thumb_path)

        # 2. [CN] Supabase Storage
        file_url = upload_to_storage("uploads", temp_path)
        thumb_url = upload_to_storage("thumbnails", thumb_path)

        # 3. AI [CN] + [CN]
        description = describe_image(temp_path)
        full_desc = description
        if extra_description:
            full_desc = f"{description}\n{extra_description}"

        embedding = get_embedding(full_desc)

        # 4. [CN] Supabase DB
        upload_time = datetime.now().isoformat()
        result = db_add_image({
            "file_name": fn,
            "file_path": file_url,
            "thumbnail_path": thumb_url,
            "uploader": uploader,
            "upload_time": upload_time,
            "description": description,
            "extra_description": extra_description,
            "source_file": fn,
            "source_url": file_url,
            "embedding": embedding,
        })

        return jsonify({
            "type": "image",
            "id": result.get("id"),
            "file_name": fn,
            "file_url": file_url,
            "thumbnail_url": thumb_url,
            "description": description,
            "uploader": uploader,
            "embedding_dim": len(embedding),
        })

    except Exception as e:
        return _err(e)


@app.route("/api/search", methods=["POST"])
def api_search():
    """_ — _"""
    body = request.get_json(silent=True) or {}
    raw_query = body.get("query", "")
    top_n = body.get("top_n", 10)

    if not raw_query:
        return jsonify({"error": "Missing query parameter"}), 400

    # 1. LLM [CN] → [CN]
    parsed = parse_query(raw_query)
    search_query = parsed["query"]

    # 2. [CN]LLM [CN] + [CN]
    filters = {}
    # [CN]
    if body.get("uploader"):
        filters["uploader"] = body["uploader"]
    elif parsed.get("uploader"):
        filters["uploader"] = parsed["uploader"]

    if body.get("date_from"):
        filters["date_from"] = body["date_from"]
    elif parsed.get("date_from"):
        filters["date_from"] = parsed["date_from"]

    if body.get("date_to"):
        filters["date_to"] = body["date_to"]
    elif parsed.get("date_to"):
        filters["date_to"] = parsed["date_to"]

    try:
        results = search_similar(search_query, top_n=top_n, filters=(filters if filters else None))

        # [CN]
        return jsonify({
            "parsed": {
                "query": search_query,
                "uploader": filters.get("uploader"),
                "date_from": filters.get("date_from"),
                "date_to": filters.get("date_to"),
            },
            "results": results,
        })
    except Exception as e:
        return _err(e)


@app.route("/api/debug", methods=["GET"])
def api_debug():
    """_"""
    results = {}
    # 1. Supabase
    try:
        from supabase_client import count_images
        results["supabase"] = f"OK ({count_images()} images)"
    except Exception as e:
        results["supabase"] = f"FAIL: {e}"
    # 2. ZhipuAI embedding
    try:
        from upload_test import get_embedding
        emb = get_embedding("test")
        results["zhipuai_embed"] = f"OK (dim={len(emb)})"
    except Exception as e:
        results["zhipuai_embed"] = f"FAIL: {e}"
    # 3. ZhipuAI vision
    try:
        resp = client.chat.completions.create(model=VISION_MODEL, messages=[{"role":"user","content":"Say hi"}])
        results["zhipuai_vision"] = "OK"
    except Exception as e:
        results["zhipuai_vision"] = f"FAIL: {e}"
    # 4. RPC
    try:
        from upload_test import get_embedding
        from supabase_client import search_similar as db_search
        emb = get_embedding("test")
        r = db_search(emb, top_n=1)
        results["pgvector_rpc"] = f"OK ({len(r)} results)"
    except Exception as e:
        results["pgvector_rpc"] = f"FAIL: {e}"
    return jsonify(results)


@app.route("/api/images", methods=["GET"])
def api_list_images():
    """_"""
    try:
        images = db_get_all_images()
        return jsonify(images)
    except Exception as e:
        return _err(e)


# ============================================================
# [CN]
# ============================================================

@app.route("/upload-web/")
def upload_web_index():
    return send_from_directory(str(BASE_DIR / "upload-web"), "index.html")

@app.route("/upload-web/<path:filename>")
def serve_upload_web(filename):
    return send_from_directory(str(BASE_DIR / "upload-web"), filename)


@app.route("/query-web/")
def query_web_index():
    return send_from_directory(str(BASE_DIR / "query-web"), "index.html")

@app.route("/query-web/<path:filename>")
def serve_query_web(filename):
    return send_from_directory(str(BASE_DIR / "query-web"), filename)


@app.route("/")
def index():
    return """
    <h1>Library Agent</h1>
    <ul>
        <li><a href="/upload-web/">Upload</a></li>
        <li><a href="/query-web/">Search</a></li>
    </ul>
    """


# ============================================================
# [CN]
# ============================================================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    # Try gunicorn first (Linux), fall back to waitress
    try:
        from gunicorn.app.base import BaseApplication
        class GunicornApp(BaseApplication):
            def __init__(self, app, port):
                self.application = app
                self.port = port
                super().__init__()
            def load_config(self):
                self.cfg.set('bind', f'0.0.0.0:{self.port}')
                self.cfg.set('workers', 1)
            def load(self):
                return self.application
        GunicornApp(app, port).run()
    except ImportError:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
