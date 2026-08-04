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

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Receive file, stash to OSS, return immediately."""
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    fn = file.filename.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    uploader = request.form.get("uploader", "test_user")
    suffix = '.' + fn.rsplit('.', 1)[-1].lower() if '.' in fn else ''

    import tempfile, hashlib
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    file.save(tmp.name)
    tmp.close()
    temp_path = Path(tmp.name)

    try:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"incoming/{ts}_{hashlib.md5(fn.encode()).hexdigest()[:8]}{suffix}"

        # Save to OSS incoming folder
        from supabase_client import _get_oss, OSS_BUCKET
        s3 = _get_oss()
        with open(temp_path, "rb") as f:
            s3.put_object(Bucket=OSS_BUCKET, Key=safe_name, Body=f.read())

        # Save metadata as a JSON sidecar
        meta = {"original_name": fn, "uploader": uploader, "time": datetime.now().isoformat()}
        s3.put_object(Bucket=OSS_BUCKET, Key=safe_name + ".meta.json",
                      Body=_json.dumps(meta, ensure_ascii=False))

        return jsonify({"type": "received", "file": fn, "status": "queued"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: temp_path.unlink()
        except: pass


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

    try:
        if suffix == ".pdf":
            from pdf_processor import process_pdf
            result = process_pdf(temp_path.as_posix(), uploader=uploader, source_name=fn)
            return jsonify({"type":"pdf","name":result.get("pdf_name",""),"page_count":result.get("page_count",0),"summary":str(result.get("summary","")),"image_count":result.get("image_count",0),"indexed_count":result.get("indexed_count",0)})
        elif suffix == ".docx":
            from docx_processor import process_docx
            result = process_docx(temp_path.as_posix(), uploader=uploader, source_name=fn)
            return jsonify({"type":"docx","name":result.get("docx_name",""),"summary":str(result.get("summary","")),"image_count":result.get("image_count",0),"indexed_count":len(result.get("image_results",[]))})
        else:
            from upload_test import process_image
            r = process_image(str(temp_path), uploader=uploader, source_file=fn)
            return jsonify({"type":"image","id":r.get("id"),"file_name":r.get("file_name"),"file_url":r.get("file_path"),"thumbnail_url":r.get("thumbnail_path"),"description":r.get("description",""),"uploader":uploader,"embedding_dim":2048})
    except Exception as e:
        import traceback as _tb2
        _tb2.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def api_search():
    """_ — _"""
    body = request.get_json(silent=True) or {}
    raw_query = body.get("query", "")
    top_n = body.get("top_n", 10)

    if not raw_query:
        return jsonify({"error": "Missing query parameter"}), 400

    # 1. DeepSeek query parsing (fast from China)
    import httpx as _httpx
    parsed = {"query": raw_query, "uploader": None, "date_from": None, "date_to": None}
    try:
        ds_resp = _httpx.post('https://api.deepseek.com/v1/chat/completions',
            headers={'Authorization': f'Bearer sk-2cd1cbfa456a4d4ca5f974d2913dd2c1', 'Content-Type': 'application/json'},
            json={'model':'deepseek-chat', 'messages':[{'role':'system','content':'你是搜索语句解析器。去掉量词(张/个/份/篇/条/款/种)、口语词(给我/我要/帮我/我想/看看/搜/找/有没有)和语气词，保留核心搜索词(空格分隔)和过滤条件。只输出JSON：{"query":"核心词","uploader":null或名字,"date_from":null或YYYY-MM-DD,"date_to":null或YYYY-MM-DD}'},{'role':'user','content': raw_query}], 'max_tokens':100},
            timeout=5)
        if ds_resp.status_code == 200:
            import json as _j2
            p = _j2.loads(ds_resp.json()['choices'][0]['message']['content'].strip())
            parsed.update(p)
    except Exception: pass
    search_query = parsed["query"]

    # 2. [CN]LLM [CN] + [CN]
    filters = {}
    if body.get("uploader"): filters["uploader"] = body["uploader"]
    elif parsed.get("uploader"): filters["uploader"] = parsed["uploader"]
    if body.get("date_from"): filters["date_from"] = body["date_from"]
    elif parsed.get("date_from"): filters["date_from"] = parsed["date_from"]
    if body.get("date_to"): filters["date_to"] = body["date_to"]
    elif parsed.get("date_to"): filters["date_to"] = parsed["date_to"]

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


@app.route("/api/dedup", methods=["POST"])
def api_dedup():
    """Run pixel-level dedup. Called by cron/GitHub Actions."""
    import subprocess, sys
    try:
        proc = subprocess.run(
            [sys.executable, str(BASE_DIR / "dedup_images.py")],
            capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        return jsonify({"ok": True, "output": proc.stdout[-500:]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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

@app.route("/library-web/")
def library_web_index():
    return send_from_directory(str(BASE_DIR / "library-web"), "index.html")

@app.route("/library-web/<path:filename>")
def serve_library_web(filename):
    return send_from_directory(str(BASE_DIR / "library-web"), filename)


@app.route("/query-web/<path:filename>")
def serve_query_web(filename):
    return send_from_directory(str(BASE_DIR / "query-web"), filename)


@app.route("/logo-small.png")
def serve_logo():
    return send_from_directory(str(BASE_DIR), "logo-small.png")


@app.route("/")
def index():
    return """
    <h1>Library Agent</h1>
    <ul>
        <li><a href="/upload-web/">Upload</a></li>
        <li><a href="/query-web/">Search</a></li>
        <li><a href="/library-web/">Library</a></li>
    </ul>
    """


# ============================================================
# [CN]
# ============================================================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    from waitress import serve
    serve(app, host="0.0.0.0", port=port, threads=2)
