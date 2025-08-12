from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

from pdf_to_excel import (
    UPLOAD_DIR, THUMB_DIR, ENHANCED_DIR,
    get_pdf_page_count, generate_page_thumbnail, process_pdf_page
)

app = Flask(__name__)
CORS(app)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)

@app.after_request
def add_no_cache_headers(resp):
    if request.path.startswith("/images/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

# Serve thumbnails & enhanced crops
@app.route("/images/thumbs/<path:filename>")
def thumbs(filename):
    return send_from_directory(THUMB_DIR, filename)

@app.route("/images/enhanced/<path:filename>")
def images_enhanced(filename):
    return send_from_directory(ENHANCED_DIR, filename)

# Upload a PDF
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a .pdf file"}), 400

    save_path = os.path.join(UPLOAD_DIR, f.filename)
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
        "filename": f.filename,
        "page_count": page_count,
        "thumbnails": thumbs
    })

# Process a specific page
@app.route("/process_page", methods=["POST"])
def process_page():
    filename = request.form.get("filename")
    page_number = request.form.get("page_number", type=int)
    if not filename:
        return jsonify({"error": "Missing filename"}), 400
    if not page_number or page_number < 1:
        return jsonify({"error": "Missing or invalid page_number"}), 400

    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": "File not found on server. Re-upload the PDF."}), 404

    data = process_pdf_page(pdf_path, page_number)
    return jsonify({"message": "Page processed", "data": data})

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
