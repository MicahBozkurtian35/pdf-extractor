from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pdf_to_excel import process_pdf  # <-- import your extractor function

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith(".pdf"):
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)

        # Run your PDF extraction logic here
        extracted_data = process_pdf(save_path)

        return jsonify({
            "message": "File processed successfully",
            "data": extracted_data
        })
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
