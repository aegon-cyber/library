"""
server.py — library-agent API 服务器

提供 REST API 给前端调用，封装 ZhipuAI + Supabase 流水线。

启动: python server.py
端口: 5050
"""

import io
import sys
import os as _os

# 强制 UTF-8，解决 Windows GBK 编码问题（必须在最前面）
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
_os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 全局替换 print：任何模块内的 print 都走 UTF-8-safe 通道
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

app = Flask(__name__, static_folder=None)
CORS(app)

# ── Windows GBK 中间件：兜底捕获编码错误 ──
@app.errorhandler(UnicodeEncodeError)
def handle_gbk_error(e):
    return jsonify({"error": "Server encoding error, please try again"}), 500


# 本地临时目录
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

THUMB_DIR = BASE_DIR / "thumbnails"
THUMB_DIR.mkdir(exist_ok=True)


# ============================================================
# 自然语言解析
# ============================================================

def parse_query(raw_input: str) -> dict:
    """用 LLM 从自然语言中提取搜索意图和过滤条件。

    例如 "京东七鲜七月份的饮料海报"
      → {query: "饮料海报", uploader: "京东七鲜", date_from: "2026-07-01", date_to: "2026-07-31"}

    如果没有提到上传人或日期，对应字段返回 None。
    """
    # 先查现有上传人名单，帮 LLM 做精确判断
    try:
        all_images = db_get_all_images()
        known_uploaders = list(set(img.get("uploader", "") for img in all_images if img.get("uploader")))
    except Exception:
        known_uploaders = []

    uploader_hint = ""
    if known_uploaders:
        uploader_hint = f"\n当前已知上传人（只有精确匹配才算过滤条件）：{_json.dumps(known_uploaders, ensure_ascii=False)}"

    prompt = f"""你是一个搜索意图解析器。从用户输入中提取以下信息，输出纯JSON：

1. query: 去掉上传人、日期等条件后的纯内容搜索词。如果没有实质内容则保留原始输入
2. uploader: 如果用户提到了和已知上传人列表精确匹配的名字则输出，否则null
3. date_from / date_to: 如果用户提到时间范围则输出 YYYY-MM-DD 格式，否则null
   - "七月" → 当年7月1日~7月31日
   - "上周" / "昨天" → 根据当前日期 {datetime.now().strftime('%Y-%m-%d')} 推算
   - "最近一个月" → 30天前到今天

⚠️ 只有明确提到上传来源时才填 uploader。像"万里长城""百里挑一"里的"万里""百里"不是人名，要填null。
{uploader_hint}

用户输入：{raw_input}

只输出JSON，不要其他文字："""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,  # 文本能力足够，轻量任务
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        # 清理可能的 markdown 代码块包裹
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
        # 解析失败就原样返回
        return {"query": raw_input, "uploader": None, "date_from": None, "date_to": None}


# ============================================================
# API 路由
# ============================================================

@app.route("/api/process", methods=["POST"])
def api_process():
    """上传文件 → AI处理 → 向量化 → Supabase存储。支持图片、PDF、DOCX。"""
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
    # ── 先提取文件 ──
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
    # 保留扩展名
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

    if 'multipart/form-data' in content_type:
        # 手动解析 multipart，跳过 Werkzeug
        import email.parser as _ep, io as _io
        from email.policy import default as _default_policy
        boundary = content_type.split('boundary=', 1)[-1].strip()
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
        boundary_bytes = boundary.encode('utf-8')

        # 按 boundary 拆分
        parts = raw_data.split(b'--' + boundary_bytes)
        for part in parts:
            if b'name="file"' in part[:200] or b'name=file' in part[:200]:
                # 提取 filename
                header_end = part.find(b'\r\n\r\n')
                if header_end == -1:
                    continue
                headers = part[:header_end].decode('utf-8', errors='replace')
                for line in headers.split('\r\n'):
                    if 'filename=' in line:
                        fn = line.split('filename=', 1)[-1].strip().strip('"')
                # 提取文件数据
                file_data = part[header_end + 4:]
                if file_data.endswith(b'\r\n'):
                    file_data = file_data[:-2]
                break
            if b'name="uploader"' in part[:250]:
                header_end = part.find(b'\r\n\r\n')
                if header_end != -1:
                    uploader = part[header_end+4:].decode('utf-8', errors='replace').strip()
                    if uploader.endswith('\r\n'):
                        uploader = uploader[:-2]

    if not file_data or file_data == raw_data:
        return jsonify({"error": "Cannot extract file content"}), 400

    if fn == '':
        return jsonify({"error": "Empty filename"}), 400

    extra_description = None  # multipart 里可能有

    suffix = ''.join(Path(fn).suffixes) if '.' in fn else ''
    if not suffix:
        suffix = '.bin'
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(file_data)
    tmp.close()
    temp_path = Path(tmp.name)

    # ── PDF 分支 ──
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
            return jsonify({"error": str(e)}), 500

    # ── DOCX 分支（子进程模式，绕开 Windows GBK 限制）──
    if suffix == ".docx":
        import subprocess, json as _json2
        # 把参数写入临时 JSON 文件，避免命令行编码问题
        args_file = TEMP_DIR / f"_args_{temp_path.stem}.json"
        args_file.write_text(_json2.dumps({
            "path": temp_path.as_posix(),
            "uploader": uploader
        }), encoding="utf-8")

        # 用独立的 Python 脚本处理
        worker_script = TEMP_DIR / "_docx_worker.py"
        worker_script.write_text(r'''
import sys, json
from pathlib import Path
BASE = Path(r"''' + str(BASE_DIR) + r'''")
sys.path.insert(0, str(BASE))
args = json.loads(Path(sys.argv[1]).read_text("utf-8"))
from docx_processor import process_docx
r = process_docx(args["path"], uploader=args["uploader"])
result = {"name":r["docx_name"],"summary":r["summary"],"image_count":r["image_count"],"indexed_count":len(r["image_results"])}
print("__OK__" + json.dumps(result, ensure_ascii=False))
''', encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(worker_script), str(args_file)],
                capture_output=True, text=True, timeout=300,
                encoding="utf-8", errors="replace",
                env={**dict(_os.environ), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            )
            out = proc.stdout
            if "__OK__" in out and proc.returncode == 0:
                data = _json2.loads(out.split("__OK__", 1)[1])
                return jsonify({"type": "docx", **data})
            else:
                return jsonify({"error": f"Worker failed: {(out + proc.stderr)[:300]}"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # 清理临时文件
            try: args_file.unlink()
            except: pass
            try: worker_script.unlink()
            except: pass

    # ── 图片分支 ──
    img_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    if suffix not in img_exts:
        return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

    try:
        # 1. 生成缩略图
        img = Image.open(temp_path)
        thumb_width = 200
        thumb_height = int(img.height * (thumb_width / img.width))
        img.thumbnail((thumb_width, thumb_height), Image.LANCZOS)
        thumb_path = THUMB_DIR / f"thumb_{temp_path.stem}.png"
        img.save(thumb_path)

        # 2. 上传到 Supabase Storage
        file_url = upload_to_storage("uploads", temp_path)
        thumb_url = upload_to_storage("thumbnails", thumb_path)

        # 3. AI 描述 + 向量化
        description = describe_image(temp_path)
        full_desc = description
        if extra_description:
            full_desc = f"{description}\n{extra_description}"

        embedding = get_embedding(full_desc)

        # 4. 存入 Supabase DB
        upload_time = datetime.now().isoformat()
        result = db_add_image({
            "file_name": safe_name,
            "file_path": file_url,
            "thumbnail_path": thumb_url,
            "uploader": uploader,
            "upload_time": upload_time,
            "description": description,
            "extra_description": extra_description,
            "source_file": safe_name,
            "source_url": file_url,
            "embedding": embedding,
        })

        return jsonify({
            "type": "image",
            "id": result.get("id"),
            "file_name": safe_name,
            "file_url": file_url,
            "thumbnail_url": thumb_url,
            "description": description,
            "uploader": uploader,
            "embedding_dim": len(embedding),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def api_search():
    """语义搜索图片 — 支持自然语言过滤"""
    body = request.get_json(silent=True) or {}
    raw_query = body.get("query", "")
    top_n = body.get("top_n", 10)

    if not raw_query:
        return jsonify({"error": "Missing query parameter"}), 400

    # 1. LLM 解析自然语言 → 提取过滤条件
    parsed = parse_query(raw_query)
    search_query = parsed["query"]

    # 2. 构建过滤器（LLM 解析结果 + 前端手动覆盖）
    filters = {}
    # 前端显式传入的优先
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

        # 标注解析结果，前端可以用
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
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug", methods=["GET"])
def api_debug():
    """测试各组件"""
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
    """获取所有已索引图片"""
    try:
        images = db_get_all_images()
        return jsonify(images)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 静态文件
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
# 启动
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Library Agent API Server")
    print("  上传: http://localhost:5050/upload-web/")
    print("  搜索: http://localhost:5050/query-web/")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
