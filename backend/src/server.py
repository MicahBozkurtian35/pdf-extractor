import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# Use the same dirs and helpers defined in PDF_Data_Extract so paths stay in sync
from PDF_Data_Extract import (
    UPLOAD_DIR,
    THUMB_DIR,
    ENHANCED_DIR,
    get_pdf_page_count,
    generate_page_thumbnail,
    process_pdf_page,
    # reset_cache,  # optional; only if you implemented it
)

ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
CORS(app)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure images are never cached by the browser
@app.after_request
def add_no_cache_headers(resp):
    p = request.path or ""
    if p.startswith("/thumbnail/") or p.startswith("/images/enhanced/") or p.startswith("/images/thumbs/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

# ---------------------------
# Static image routes
# ---------------------------

# Current thumbnails route (used by the updated frontend)
@app.route("/thumbnail/<path:filename>")
def get_thumbnail(filename):
    return send_from_directory(THUMB_DIR, filename)

# Back-compat thumbnail route (used by your older frontend)
@app.route("/images/thumbs/<path:filename>")
def get_thumbs_legacy(filename):
    return send_from_directory(THUMB_DIR, filename)

# ✅ Enhanced crops route (what you’re missing)
@app.route("/images/enhanced/<path:filename>")
def get_enhanced_image(filename):
    return send_from_directory(ENHANCED_DIR, filename)

# ---------------------------
# Upload & process
# ---------------------------

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Please upload a .pdf file"}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)

    try:
        page_count = get_pdf_page_count(save_path)
        thumbs = []
        for p in range(1, page_count + 1):
            thumbs.append(generate_page_thumbnail(save_path, p))
    except Exception as e:
        return jsonify({"error": f"Failed to prepare PDF: {e}"}), 500

    return jsonify({
        "message": "PDF uploaded",
        "filename": filename,
        "page_count": page_count,
        "thumbnails": thumbs
    })

# The updated frontend calls this with JSON: { "filename": "...", "page": <int> }
@app.route("/process_page", methods=["POST"])
def process_page():
    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename")
    page_number = payload.get("page", None)

    if not filename:
        return jsonify({"error": "Missing filename"}), 400
    if page_number is None:
        return jsonify({"error": "Missing page"}), 400
    try:
        page_number = int(page_number)
        if page_number < 1:
            raise ValueError
    except Exception:
        return jsonify({"error": "Invalid page"}), 400

    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": "File not found on server. Re-upload the PDF."}), 404

    data = process_pdf_page(pdf_path, page_number)
    return jsonify({"message": "Page processed", "data": data})

# Optional reset endpoint if you implemented reset_cache() in PDF_Data_Extract.py
# @app.route("/reset", methods=["POST"])
# def reset():
#     reset_cache()
#     return jsonify({"message": "Cache reset"})

if __name__ == "__main__":
    # Bind to localhost; change host to "0.0.0.0" for LAN access if needed.
    app.run(host="localhost", port=5000, debug=True)
