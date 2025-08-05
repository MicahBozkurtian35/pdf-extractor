import os
import requests
import re
import json
import base64
import io
import tempfile
import anthropic
import pdfplumber
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Configuration
PDF_PATH = r"C:\Users\Mbomm\IdeaProjects\PDF Graph Scanner\input_pdfs\Sample3.pdf"
UPSTAGE_API_KEY_PATH = r"C:\API\upstage_api_key.txt"
ANTHROPIC_API_KEY_PATH = r"C:\API\anthropic_key.txt"
OUTPUT_DIR = "pdf_extraction_output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def read_upstage_api_key(file_path=UPSTAGE_API_KEY_PATH):
    """Reads the Upstage API key from a file"""
    with open(file_path, "r") as f:
        return f.read().strip()

def read_anthropic_key(file_path=ANTHROPIC_API_KEY_PATH):
    """Reads the Anthropic API key from a file"""
    with open(file_path, "r") as f:
        return f.read().strip()

def call_upstage_api(pdf_path):
    """Call the Upstage API to get figure bounding boxes"""
    api_key = read_upstage_api_key()

    url = "https://api.upstage.ai/v1/document-ai/document-parse"
    params = {
        "ocr": "force",
        "base64_encoding": "['table']",
        "model": "document-parse"
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Calling Upstage API for {pdf_path}...")
    with open(pdf_path, "rb") as f:
        files = {"document": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(url, headers=headers, files=files, data=params)

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None

    result = response.json()

    # Save raw API response for reference
    with open(os.path.join(OUTPUT_DIR, "upstage_response.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result

def extract_figure_boxes(upstage_response):
    """Extract figure bounding boxes from the Upstage response"""
    if not upstage_response or "content" not in upstage_response or "html" not in upstage_response["content"]:
        return []

    html_content = upstage_response["content"]["html"]

    # Extract figure elements from HTML using regex
    figure_pattern = r'<figure id=[\'"](.*?)[\'"] data-category=[\'"](chart|table|figure)[\'"]><img data-coord="top-left:\((\d+),(\d+)\); bottom-right:\((\d+),(\d+)\)"'
    matches = re.findall(figure_pattern, html_content)

    figures = []
    for i, match in enumerate(matches):
        fig_id, fig_type, left, top, right, bottom = match

        # Skip tables if you only want charts
        if fig_type.lower() == 'table':
            continue

        # Convert to integers
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)

        figures.append({
            "id": fig_id,
            "type": fig_type,
            "page": 1,  # Default to page 1, adjust if needed
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": right - left,
            "height": bottom - top,
            "original_index": i
        })

    print(f"Found {len(figures)} figures/charts")
    return figures

def crop_pdf_with_pdfplumber(pdf_path, figure, expand_margin=5):
    """
    Extracts a chart region from the PDF using pdfplumber.
    Returns a new PDF containing only the cropped region.
    """
    try:
        page_num = figure.get("page", 1) - 1  # Convert to 0-based index

        # Get bbox coordinates (adjust if needed)
        x0 = figure["left"] - expand_margin
        y0 = figure["top"] - expand_margin
        x1 = figure["right"] + expand_margin
        y1 = figure["bottom"] + expand_margin

        # Ensure coordinates are non-negative
        x0 = max(0, x0)
        y0 = max(0, y0)

        # Open the PDF with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                print(f"Error: Page {page_num+1} does not exist in the PDF")
                return None

            page = pdf.pages[page_num]

            # Get page dimensions to ensure we don't exceed them
            page_width = page.width
            page_height = page.height

            # Ensure coordinates don't exceed page dimensions
            x1 = min(x1, page_width)
            y1 = min(y1, page_height)

            # Extract the cropped region
            crop = page.crop((x0, y0, x1, y1))

            # Create a new PDF with the cropped content
            output_path = os.path.join(OUTPUT_DIR, f"crop_fig_{figure['id']}.pdf")

            # Save the cropped region to a new PDF
            # Note: pdfplumber doesn't have a direct way to save a crop as PDF
            # We'll use its to_image() to render it and save that as PDF

            # Method 1: Create a PDF with the cropped image
            img = crop.to_image(resolution=150)  # Adjust resolution as needed

            # Create a temporary file to hold the image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                img.save(tmp_path)

            # Create a PDF from the image
            c = canvas.Canvas(output_path, pagesize=(crop.width, crop.height))
            c.drawImage(tmp_path, 0, 0, width=crop.width, height=crop.height)
            c.save()

            # Clean up the temporary file
            os.unlink(tmp_path)

            debug_print(f"Saved cropped chart to {output_path}")
            return output_path

    except Exception as e:
        print(f"Error cropping PDF: {e}")
        return None

def extract_with_pypdf2(pdf_path, figure, expand_margin=5):
    """
    Alternative method using PyPDF2 to extract a chart region from the PDF.
    This preserves vector content but may not always crop perfectly.
    """
    try:
        page_num = figure.get("page", 1) - 1  # Convert to 0-based index

        # Open the PDF
        reader = PdfReader(pdf_path)

        if page_num >= len(reader.pages):
            print(f"Error: Page {page_num+1} does not exist in the PDF")
            return None

        # Get the page
        page = reader.pages[page_num]

        # Get page dimensions
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)

        # Convert coordinates to PDF units (from bottom-left origin)
        # PyPDF2 uses bottom-left as origin, but Upstage likely uses top-left
        x0 = figure["left"] - expand_margin
        y0 = page_height - (figure["bottom"] + expand_margin)  # Convert to bottom-left origin
        x1 = figure["right"] + expand_margin
        y1 = page_height - (figure["top"] - expand_margin)  # Convert to bottom-left origin

        # Ensure coordinates are within page
        x0 = max(0, min(x0, page_width))
        y0 = max(0, min(y0, page_height))
        x1 = max(0, min(x1, page_width))
        y1 = max(0, min(y1, page_height))

        # Set the crop box
        page.cropbox.lower_left = (x0, y0)
        page.cropbox.upper_right = (x1, y1)

        # Create a new PDF with the cropped page
        writer = PdfWriter()
        writer.add_page(page)

        # Save the result
        output_path = os.path.join(OUTPUT_DIR, f"crop_pypdf_fig_{figure['id']}.pdf")
        with open(output_path, "wb") as f:
            writer.write(f)

        debug_print(f"Saved PyPDF2 cropped chart to {output_path}")
        return output_path

    except Exception as e:
        print(f"Error in PyPDF2 extraction: {e}")
        return None

def analyze_with_claude(pdf_path, anthropic_key):
    """
    Send the PDF to Claude for analysis and extract chart data
    """
    # Read the PDF file
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    # Convert to base64
    pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")

    # Create the Claude client
    client = anthropic.Anthropic(api_key=anthropic_key)

    # Prepare the prompt for Claude
    prompt = """
Please analyze this PDF chart image and extract the data with high precision.

For each chart:
1. Identify the chart type (Line, Bar, Pie, etc.)
2. Extract all data points with exact values
3. Identify axes labels and scales
4. For line charts, capture all significant points including peaks and valleys
5. For bar charts, measure the exact height of each bar

Return the data in CSV format along with chart metadata in this JSON structure:
{
  "chart_type": "Line Chart",
  "data_representation": {"Line series": ["Series1", "Series2"]},
  "csv_data": "X,Series1,Series2\\n2020,15.2,18.7\\n2021,16.8,19.2\\n...",
  "axes_info": {
    "x_axis": {"label": "Year", "type": "continuous"},
    "y_axis": {"label": "Value", "min": 0, "max": 50}
  }
}

Be extremely precise with measurements. If you're unsure about exact values, provide your best estimate based on the chart scale.
    """

    try:
        # Call the Claude API
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4096,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}}
                    ]
                }
            ]
        )

        # Get the response text
        result = response.content[0].text

        # Try to extract JSON from the response
        try:
            # Look for JSON content using regex
            json_pattern = r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})'
            json_match = re.search(json_pattern, result)

            if json_match:
                # Get the match that worked (either group 1 or 2)
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
                try:
                    # Clean up potential trailing commas which are invalid in JSON
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    chart_data = json.loads(json_str)
                    return chart_data
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from Claude response: {e}")
                    # Save the raw response for debugging
                    with open(os.path.join(OUTPUT_DIR, f"claude_response_error_{os.path.basename(pdf_path)}.txt"), "w") as f:
                        f.write(result)
                    return {"error": "JSON parsing error", "raw_response": result}
            else:
                # No JSON found, return the raw response
                return {"error": "No JSON found in response", "raw_response": result}

        except Exception as e:
            print(f"Error extracting JSON from Claude response: {e}")
            return {"error": str(e), "raw_response": result}

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return {"error": str(e)}

def save_csv_data(chart_data, figure_id):
    """Save extracted CSV data to a file"""
    if not chart_data or "csv_data" not in chart_data:
        print(f"No CSV data found for figure {figure_id}")
        return

    csv_path = os.path.join(OUTPUT_DIR, f"data_fig_{figure_id}.csv")
    with open(csv_path, "w") as f:
        f.write(chart_data["csv_data"])

    print(f"Saved CSV data to {csv_path}")

def save_results_to_json(results):
    """Save all extraction results to a JSON file"""
    output_path = os.path.join(OUTPUT_DIR, "extraction_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved complete results to {output_path}")

def process_pdf(pdf_path):
    """Main processing function"""
    # 1. Call Upstage API to get chart locations
    upstage_response = call_upstage_api(pdf_path)
    if not upstage_response:
        print("Failed to get response from Upstage API")
        return

    # 2. Extract figure bounding boxes
    figures = extract_figure_boxes(upstage_response)
    if not figures:
        print("No figures/charts found")
        return

    # 3. Read Claude API key
    anthropic_key = read_anthropic_key()
    if not anthropic_key:
        print("Claude API key not found")
        return

    # 4. Process each figure
    results = []

    for figure in figures:
        print(f"\nProcessing figure {figure['id']} ({figure['type']})...")

        # Try both PDF extraction methods
        crop_path_pdfplumber = crop_pdf_with_pdfplumber(pdf_path, figure)
        crop_path_pypdf = extract_with_pypdf2(pdf_path, figure)

        # Use pdfplumber result as default, fall back to PyPDF2 if needed
        crop_path = crop_path_pdfplumber if crop_path_pdfplumber else crop_path_pypdf

        if not crop_path:
            print(f"Failed to crop figure {figure['id']}")
            continue

        # 5. Analyze with Claude
        print(f"Sending cropped PDF to Claude for analysis...")
        chart_data = analyze_with_claude(crop_path, anthropic_key)

        # 6. Save CSV data
        if chart_data and "csv_data" in chart_data:
            save_csv_data(chart_data, figure['id'])

        # 7. Store results
        results.append({
            "figure": figure,
            "crop_path": crop_path,
            "chart_data": chart_data
        })

    # 8. Save all results
    save_results_to_json(results)

    print("\nProcessing complete!")
    print(f"Check the '{OUTPUT_DIR}' folder for results.")

    return results

if __name__ == "__main__":
    process_pdf(PDF_PATH)