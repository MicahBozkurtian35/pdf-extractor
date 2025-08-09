from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pdf_to_excel import process_pdf_to_data
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "enhanced_images")

@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(IMAGES_DIR, filename)


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
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Call the modified function that returns structured data
        extracted_data = process_pdf_to_data(file_path)
        return jsonify({
            "message": "File processed successfully",
            "data": extracted_data,
        })
    else:
        return jsonify({"error": "Invalid file type"}), 400


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
