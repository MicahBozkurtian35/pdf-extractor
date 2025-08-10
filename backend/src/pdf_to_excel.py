import os
import io
import re
import base64
import csv
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import requests
from dotenv import load_dotenv

# Load .env (must be in same folder as server.py/pdf_to_excel.py)
load_dotenv()

# --------- helpers ----------
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

# -----------------------------
# Step 1: PDF -> PNG images
# -----------------------------
def pdf_to_pngs(pdf_path, output_dir="temp_images"):
    os.makedirs(output_dir, exist_ok=True)
    poppler_path = os.getenv("POPPLER_PATH") or None  # if Poppler is on PATH, leave None

    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    image_paths = []
    for i, image in enumerate(images):
        box = image.getbbox()
        if not box:
            continue

        x0, y0, x1, y1 = box
        if (x1 - x0) > 200 and (y1 - y0) > 100:
            graph_image = image.crop(box)
            out_path = os.path.join(output_dir, f"chart_{i}.png")
            graph_image.save(out_path, "PNG")
            image_paths.append(out_path)

    return image_paths

# -----------------------------
# Step 2: Enhance image (zoom + optional grid)
# -----------------------------
def enhance_image(image_path, output_dir="enhanced_images"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        img = Image.open(image_path)

        upscale = _env_float("UPSCALE_FACTOR", 1.2)
        draw_grid = _env_bool("GRID_OVERLAY", False)
        grid_step = _env_int("GRID_SPACING", 25)

        if upscale and abs(upscale - 1.0) > 1e-6:
            img = img.resize(
                (int(img.width * upscale), int(img.height * upscale)),
                Image.LANCZOS
            )

        if draw_grid and grid_step > 0:
            draw = ImageDraw.Draw(img)
            for x in range(0, img.width, grid_step):
                draw.line([(x, 0), (x, img.height)], fill="grey")
            for y in range(0, img.height, grid_step):
                draw.line([(0, y), (img.width, y)], fill="grey")

        out_path = os.path.join(output_dir, os.path.basename(image_path))
        img.save(out_path, "PNG")
        return out_path
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return None

# -----------------------------
# Step 3: Call OpenRouter (VISION-capable model)
# -----------------------------
def extract_data_with_openrouter(image_path: str, model: str | None = None) -> str | None:
    """
    Sends the enhanced image to OpenRouter with a strict CSV-only instruction.
    Returns raw string (ideally CSV). Your CSV parser handles fences/TSV fallback.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        return None

    # Single env knob to switch models freely on OpenRouter
    model_id = model or os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL", "openrouter/auto")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # optional attribution headers recommended by OpenRouter
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "pdf-extractor"),
    }

    # Read image as base64 data URL
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image for OpenRouter: {e}")
        return None

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a strict CSV extractor. From the chart image, extract any tabular/series data. "
                            "Respond with CSV only. No commentary, no code fences, no backticks, no markdown."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    },
                ],
            }
        ],
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
    except Exception as e:
        print(f"OpenRouter network error: {e}")
        return None

    if not resp.ok:
        print(f"OpenRouter HTTP error {resp.status_code}: {resp.text[:400]}")
        return None

    try:
        data = resp.json()
        print("DEBUG: Full OpenRouter JSON response:", data)
        content = data["choices"][0]["message"]["content"]
        return content.strip() if content else None
    except Exception:
        print(f"Unexpected OpenRouter response: {resp.text[:400]}")
        return None

# -----------------------------
# Step 4: CSV text -> DataFrame
# -----------------------------
def convert_csv_to_dataframe(raw_text: str):
    """Parse CSV/TSV-ish text robustly into a DataFrame."""
    try:
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()

        # Strip code fences if a model ignored instructions
        if text.startswith("```"):
            text = re.sub(r"^```(?:csv|tsv)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # Sniff delimiter
        sample = "\n".join(text.splitlines()[:10])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delim = dialect.delimiter
        except Exception:
            if "\t" in sample:
                delim = "\t"
            elif "," in sample:
                delim = ","
            else:
                # Convert 2+ spaces to comma as a fallback
                text = re.sub(r"[ ]{2,}", ",", text)
                delim = ","

        if delim == ",":
            text = text.replace("\t", ",")

        df = pd.read_csv(io.StringIO(text), delimiter=delim)
        return None if df.empty else df
    except Exception as e:
        print(f"Error converting text to DataFrame: {e}")
        return None

# -----------------------------
# Main pipeline for Flask
# -----------------------------
def process_pdf_to_data(pdf_path):
    try:
        temp_images = pdf_to_pngs(pdf_path)
        results = []
        debug_raw = []  # optional: helpful for debugging

        for image_path in temp_images:
            enhanced_path = enhance_image(image_path)
            if not enhanced_path:
                print(f"Skipping enhancement: {image_path}")
                continue

            print(f"Processing image: {enhanced_path}")
            extracted_text = extract_data_with_openrouter(enhanced_path)
            print("Raw model output (first 300 chars):", (extracted_text or "")[:300])

            debug_raw.append({
                "image": os.path.basename(image_path),
                "raw": (extracted_text or "")[:1000]
            })

            if extracted_text:
                df = convert_csv_to_dataframe(extracted_text)
                if df is not None:
                    results.append({
                        "image": os.path.basename(image_path),
                        "data": df.to_dict(orient="records")
                    })

        # Return both parsed tables and raw text (raw is useful while iterating)
        return {
            "tables": results,
            "debug_raw": debug_raw
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Optional local test
# -----------------------------
if __name__ == "__main__":
    test_pdf = r"C:\Users\Mbomm\IdeaProjects\PDF Graph Scanner\input_pdfs\Sample1.pdf"
    print(process_pdf_to_data(test_pdf))
