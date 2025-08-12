# document_parse_tester.py
import os
import io
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# Optional local fallback uses Pillow/pdf2image
try:
    from PIL import Image, UnidentifiedImageError  # noqa
except Exception:
    Image = None
    UnidentifiedImageError = Exception

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "")
PORT = int(os.getenv("UPSTAGE_TEST_PORT", "5001"))
ALLOWED_ORIGIN = os.getenv("UPSTAGE_TEST_ALLOWED_ORIGIN", "*")

# Upstage (per docs screenshot)
UPSTAGE_URL = "https://api.upstage.ai/v1/document-digitization"
UPSTAGE_MODEL = "document-parse"  # required for document parsing

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

os.makedirs("debug_uploads", exist_ok=True)
os.makedirs("debug_pages", exist_ok=True)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "document-parse-tester",
        "time": datetime.utcnow().isoformat(),
        "port": PORT
    }


@app.post("/parse")
def parse_document():
    import io, os, requests
    from flask import jsonify, request

    upload_field = "document" if "document" in request.files else "file"
    if upload_field not in request.files:
        return jsonify({"error": "missing form field 'document' or 'file'"}), 400

    f = request.files[upload_field]
    raw = f.read()

    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}
    files = {"document": (f.filename, io.BytesIO(raw))}
    data = {
        "ocr": "force",
        "model": "document-parse",   # <-- KEY must literally be "model"
        "return_format": "json"
    }
    print("DEBUG outgoing fields:", data)  # verify key names

    r = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    print("UPSTAGE DEBUG:", r.status_code, r.text[:300])
    try:
        r.raise_for_status()
    except Exception:
        return jsonify({"error": "Upstage HTTP error", "status": r.status_code, "text": r.text}), 502

    return jsonify(r.json())



def _run_local_mode(filename: str, raw: bytes, stamp: str):
    """
    LOCAL fallback:
    - If image: save as PNG
    - If PDF: render pages to PNG (requires pdf2image + poppler)
    """
    run_dir = os.path.join("debug_pages", f"run-{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Try as image first
    if Image:
        try:
            img = Image.open(io.BytesIO(raw))
            out_path = os.path.join(run_dir, "image-000.png")
            img.save(out_path)
            return {
                "message": "LOCAL mode (image saved)",
                "filename": filename,
                "pages": [{"type": "image", "path": out_path}],
                "mock_upstage": True
            }
        except UnidentifiedImageError:
            pass

    # Try PDF render
    try:
        from pdf2image import convert_from_bytes
        pil_pages = convert_from_bytes(raw, dpi=200)
        pages_info = []
        for i, p in enumerate(pil_pages):
            out_path = os.path.join(run_dir, f"page-{i+1:03d}.png")
            p.save(out_path)
            pages_info.append({"type": "pdf_page", "index": i + 1, "path": out_path})
        return {
            "message": "LOCAL mode (pdf pages rendered)",
            "filename": filename,
            "pages": pages_info,
            "mock_upstage": True
        }
    except Exception as e:
        return {
            "error": "LOCAL mode failed to render PDF",
            "filename": filename,
            "exception": str(e),
            "mock_upstage": True
        }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=True)
