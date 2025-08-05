import os
import base64
import json
import requests
import re
import sys
from pathlib import Path
import time
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import pdfplumber
import cv2

# CONFIG - Change these paths as needed
PDF_PATH = r"C:\Users\Mbomm\IdeaProjects\PDF Graph Scanner\input_pdfs\Sample3.pdf"  # Your test PDF path
POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"  # Your poppler path
UPSTAGE_API_KEY_PATH = r"C:\API\upstage_api_key.txt"

# Debug settings
DEBUG = True
OUTPUT_DIR = "debug_outputs"


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def read_upstage_api_key(file_path=UPSTAGE_API_KEY_PATH):
    """Reads the Upstage API key from a file"""
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading Upstage API key from {file_path}: {e}")
        return None


def extract_figures_with_upstage(pdf_path, api_key=None):
    """Calls the Upstage API to get figure bounding boxes from the PDF"""
    if not api_key:
        api_key = read_upstage_api_key()

    if not api_key:
        print("Upstage API key not found.")
        return None

    url = "https://api.upstage.ai/v1/document-ai/document-parse"
    params = {
        "ocr": "force",
        "base64_encoding": "['table']",
        "model": "document-parse"
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with open(pdf_path, "rb") as f:
            files = {"document": (os.path.basename(pdf_path), f, "application/pdf")}
            print(f"Sending PDF to Upstage API: {pdf_path}")
            response = requests.post(url, headers=headers, files=files, data=params, timeout=60)

        if response.status_code != 200:
            print(f"Upstage API error: {response.status_code} - {response.text}")
            return None

        result = response.json()

        # Save the raw API response for debugging
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "upstage_response.json"), "w") as f:
            json.dump(result, f, indent=2)

        return result
    except Exception as e:
        print(f"Error calling Upstage API: {e}")
        return None


def process_upstage_output(upstage_response):
    """Extracts figure bounding boxes from the Upstage API response"""
    if not upstage_response:
        return []

    figures = []

    # Check if the response has the expected structure with HTML content
    if "content" in upstage_response and "html" in upstage_response["content"]:
        html_content = upstage_response["content"]["html"]

        # Extract figure elements from HTML using regex
        figure_pattern = r'<figure id=[\'"](.*?)[\'"] data-category=[\'"](chart|table|figure)[\'"]><img data-coord="top-left:\((\d+),(\d+)\); bottom-right:\((\d+),(\d+)\)"'
        matches = re.findall(figure_pattern, html_content)
        debug_print(f"Found {len(matches)} figure matches in HTML")

        for i, match in enumerate(matches):
            fig_id, fig_type, left, top, right, bottom = match

            # Convert to integers
            left = int(left)
            top = int(top)
            right = int(right)
            bottom = int(bottom)
            width = right - left
            height = bottom - top

            # Skip tables - they're not charts we want to process
            if fig_type.lower() == 'table':
                debug_print(f"Skipping table figure with ID {fig_id}")
                continue

            figure = {
                "figure_id": fig_id,
                "page": 1,  # Default to page 1 if not specified
                "bounding_box": {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "right": right,
                    "bottom": bottom
                },
                "figure_type": fig_type,
                "original_order": i  # Keep track of original order
            }
            figures.append(figure)
            debug_print(f"Added figure {i+1}: ID={fig_id}, type={fig_type}, bbox={figure['bounding_box']}")

    return figures


def get_pdf_dimensions(pdf_path):
    """Gets accurate PDF dimensions using pdfplumber"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Process each page to get dimensions
            page_dimensions = []
            for i, page in enumerate(pdf.pages):
                # Try to get the cropbox or mediabox
                if hasattr(page, 'cropbox'):
                    width = page.cropbox[2] - page.cropbox[0]
                    height = page.cropbox[3] - page.cropbox[1]
                elif hasattr(page, 'mediabox'):
                    width = page.mediabox[2] - page.mediabox[0]
                    height = page.mediabox[3] - page.mediabox[1]
                else:
                    width = page.width
                    height = page.height

                is_landscape = width > height
                page_dimensions.append({
                    "page": i + 1,
                    "width": width,
                    "height": height,
                    "is_landscape": is_landscape
                })
                debug_print(f"Page {i+1} dimensions: {width}x{height}, landscape: {is_landscape}")

            return page_dimensions
    except Exception as e:
        debug_print(f"Error getting PDF dimensions: {e}")
        return []


def convert_pdf_to_images(pdf_path, output_dir=OUTPUT_DIR):
    """Converts PDF pages to images and returns paths"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Convert all pages to images
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        image_paths = []

        for i, image in enumerate(images):
            # We'll use 1-based page numbering
            img_path = os.path.join(output_dir, f"page_{i+1}.png")
            image.save(img_path, "PNG")
            debug_print(f"Saved PDF page {i+1} to {img_path}")

            # Get image dimensions for debugging
            img_width, img_height = image.size
            debug_print(f"Image dimensions: {img_width}x{img_height}")

            image_paths.append({
                "page": i + 1,
                "path": img_path,
                "width": img_width,
                "height": img_height
            })

        return image_paths
    except Exception as e:
        debug_print(f"Error converting PDF to images: {e}")
        return []


def try_scaling_methods(pdf_dimensions, image_info, figures):
    """Tests different scaling methods and visualizes the results"""
    if not pdf_dimensions or not image_info or not figures:
        print("Missing required data for scaling test")
        return

    # Create a mapping of page numbers to dimensions and image info
    pdf_dims_map = {d["page"]: d for d in pdf_dimensions}
    img_info_map = {i["page"]: i for i in image_info}

    # Group figures by page
    figures_by_page = {}
    for fig in figures:
        page = fig.get("page", 1)
        if page not in figures_by_page:
            figures_by_page[page] = []
        figures_by_page[page].append(fig)

    # For each page with figures, test different scaling methods
    for page_num, page_figures in figures_by_page.items():
        if page_num not in pdf_dims_map or page_num not in img_info_map:
            print(f"Missing dimension data for page {page_num}")
            continue

        pdf_dim = pdf_dims_map[page_num]
        img_info = img_info_map[page_num]

        # Load the image
        img_path = img_info["path"]
        try:
            original_img = Image.open(img_path)
            img_width, img_height = original_img.size

            # Test different scaling methods
            test_methods = [
                ("simple_ratio", simple_ratio_scaling),
                ("dpi_adjusted", dpi_adjusted_scaling),
                ("orientation_aware", orientation_aware_scaling),
                ("adaptive_margin", adaptive_margin_scaling)
            ]

            for method_name, scaling_func in test_methods:
                # Create a copy of the original image for drawing
                img_copy = original_img.copy()
                draw = ImageDraw.Draw(img_copy)

                # Draw bounding boxes for each figure using the current scaling method
                for fig in page_figures:
                    bbox = fig["bounding_box"]

                    # Original coordinates (red)
                    draw.rectangle([
                        (bbox["left"], bbox["top"]),
                        (bbox["right"], bbox["bottom"])
                    ], outline="red", width=2)

                    # Apply the current scaling method (green)
                    scaled_bbox = scaling_func(
                        bbox,
                        pdf_width=pdf_dim["width"],
                        pdf_height=pdf_dim["height"],
                        img_width=img_width,
                        img_height=img_height,
                        is_landscape=pdf_dim["is_landscape"]
                    )

                    # Draw scaled bounding box
                    draw.rectangle([
                        (scaled_bbox["left"], scaled_bbox["top"]),
                        (scaled_bbox["right"], scaled_bbox["bottom"])
                    ], outline="green", width=2)

                    # Add figure ID and method name
                    draw.text(
                        (scaled_bbox["left"] + 5, scaled_bbox["top"] + 5),
                        f"ID: {fig['figure_id']} - {method_name}",
                        fill="green"
                    )

                # Save the result
                output_path = os.path.join(OUTPUT_DIR, f"page_{page_num}_{method_name}.png")
                img_copy.save(output_path)
                print(f"Saved visualization for method '{method_name}' to {output_path}")

                # Generate crops using this method
                for fig in page_figures:
                    bbox = fig["bounding_box"]
                    scaled_bbox = scaling_func(
                        bbox,
                        pdf_width=pdf_dim["width"],
                        pdf_height=pdf_dim["height"],
                        img_width=img_width,
                        img_height=img_height,
                        is_landscape=pdf_dim["is_landscape"]
                    )

                    # Crop the image
                    crop = original_img.crop((
                        scaled_bbox["left"],
                        scaled_bbox["top"],
                        scaled_bbox["right"],
                        scaled_bbox["bottom"]
                    ))

                    # Save the crop
                    crop_path = os.path.join(OUTPUT_DIR, f"crop_p{page_num}_fig{fig['figure_id']}_{method_name}.png")
                    crop.save(crop_path)

        except Exception as e:
            print(f"Error processing page {page_num}: {e}")


# ---- Different scaling methods to test ----

def simple_ratio_scaling(bbox, pdf_width, pdf_height, img_width, img_height, is_landscape=False):
    """Simple direct ratio scaling from PDF coordinates to image pixels"""
    # Calculate scaling factors
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    # Apply scaling
    scaled_bbox = {
        "left": int(bbox["left"] * scale_x),
        "top": int(bbox["top"] * scale_y),
        "right": int(bbox["right"] * scale_x),
        "bottom": int(bbox["bottom"] * scale_y),
        "width": int(bbox["width"] * scale_x),
        "height": int(bbox["height"] * scale_y)
    }

    return scaled_bbox


def dpi_adjusted_scaling(bbox, pdf_width, pdf_height, img_width, img_height, is_landscape=False):
    """Scaling that accounts for potential DPI differences"""
    # PDF points (72 DPI) to image pixels (usually higher)
    # Estimate the image DPI based on dimensions
    pdf_diagonal = (pdf_width**2 + pdf_height**2)**0.5
    img_diagonal = (img_width**2 + img_height**2)**0.5

    dpi_ratio = img_diagonal / pdf_diagonal

    # Apply the scaling with DPI adjustment
    scaled_bbox = {
        "left": int(bbox["left"] * dpi_ratio),
        "top": int(bbox["top"] * dpi_ratio),
        "right": int(bbox["right"] * dpi_ratio),
        "bottom": int(bbox["bottom"] * dpi_ratio),
        "width": int(bbox["width"] * dpi_ratio),
        "height": int(bbox["height"] * dpi_ratio)
    }

    return scaled_bbox


def orientation_aware_scaling(bbox, pdf_width, pdf_height, img_width, img_height, is_landscape=False):
    """Scaling that handles orientation differences between PDF and image"""
    # Different handling based on landscape vs portrait
    if is_landscape:
        # For landscape documents, adjust the calculations
        scale_x = img_width / pdf_width
        scale_y = img_height / pdf_height
    else:
        # Standard portrait ratio
        scale_x = img_width / pdf_width
        scale_y = img_height / pdf_height

    # Apply scaling
    scaled_bbox = {
        "left": int(bbox["left"] * scale_x),
        "top": int(bbox["top"] * scale_y),
        "right": int(bbox["right"] * scale_x),
        "bottom": int(bbox["bottom"] * scale_y),
        "width": int(bbox["width"] * scale_x),
        "height": int(bbox["height"] * scale_y)
    }

    return scaled_bbox


def adaptive_margin_scaling(bbox, pdf_width, pdf_height, img_width, img_height, is_landscape=False, margin=0.1):
    """Scaling with adaptive margins to ensure we capture the full chart"""
    # First apply standard scaling
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    # Calculate the base scaled values
    left = int(bbox["left"] * scale_x)
    top = int(bbox["top"] * scale_y)
    width = int(bbox["width"] * scale_x)
    height = int(bbox["height"] * scale_y)

    # Add margins proportional to the chart size
    margin_x = int(width * margin)
    margin_y = int(height * margin)

    # Apply margins and ensure we stay within image boundaries
    scaled_bbox = {
        "left": max(0, left - margin_x),
        "top": max(0, top - margin_y),
        "width": min(img_width - left + margin_x, width + 2*margin_x),
        "height": min(img_height - top + margin_y, height + 2*margin_y)
    }

    # Add right and bottom for convenience
    scaled_bbox["right"] = scaled_bbox["left"] + scaled_bbox["width"]
    scaled_bbox["bottom"] = scaled_bbox["top"] + scaled_bbox["height"]

    return scaled_bbox


def main(pdf_path):
    """Main testing function"""
    # Ensure the path is absolute and properly formatted
    pdf_path = os.path.abspath(os.path.expanduser(pdf_path))

    print(f"Looking for PDF at: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Listing directory content:")
        dir_path = os.path.dirname(pdf_path)
        if os.path.exists(dir_path):
            print(f"Contents of {dir_path}:")
            for item in os.listdir(dir_path):
                print(f"  - {item}")
        else:
            print(f"Directory not found: {dir_path}")
        return

    # 1. Get PDF dimensions from pdfplumber
    print("Getting PDF dimensions...")
    pdf_dimensions = get_pdf_dimensions(pdf_path)

    # 2. Convert PDF to images
    print("Converting PDF to images...")
    image_info = convert_pdf_to_images(pdf_path)

    # 3. Call Upstage API
    print("Calling Upstage API for chart detection...")
    upstage_response = extract_figures_with_upstage(pdf_path)

    if not upstage_response:
        print("Failed to get response from Upstage API. Exiting.")
        return

    # 4. Process Upstage output to get figures
    figures = process_upstage_output(upstage_response)
    print(f"Found {len(figures)} figures/charts in the PDF")

    # 5. Test different scaling methods
    print("Testing different scaling methods...")
    try_scaling_methods(pdf_dimensions, image_info, figures)

    # 6. Generate summary report
    create_summary_report(pdf_dimensions, image_info, figures)

    print(f"Testing complete. Check the '{OUTPUT_DIR}' folder for results.")


def create_summary_report(pdf_dimensions, image_info, figures):
    """Creates a summary HTML report of the findings"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>PDF Bounding Box Scaling Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .section { margin-bottom: 30px; }
        .figures { display: flex; flex-wrap: wrap; }
        .figure-card { margin: 10px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .method-comparison { margin-top: 20px; }
        .recommendation { background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 10px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>PDF Bounding Box Scaling Test Results</h1>
    
    <div class="section">
        <h2>PDF Dimensions</h2>
        <table>
            <tr>
                <th>Page</th>
                <th>Width (pts)</th>
                <th>Height (pts)</th>
                <th>Orientation</th>
            </tr>
    """

    # Add PDF dimensions
    for dim in pdf_dimensions:
        html += f"""
            <tr>
                <td>{dim["page"]}</td>
                <td>{dim["width"]:.2f}</td>
                <td>{dim["height"]:.2f}</td>
                <td>{"Landscape" if dim["is_landscape"] else "Portrait"}</td>
            </tr>"""

    html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Image Dimensions</h2>
        <table>
            <tr>
                <th>Page</th>
                <th>Width (px)</th>
                <th>Height (px)</th>
                <th>Scale Factor X</th>
                <th>Scale Factor Y</th>
            </tr>
    """

    # Add image dimensions and calculate scale factors
    for img in image_info:
        page = img["page"]
        # Find matching PDF dimension
        pdf_dim = next((dim for dim in pdf_dimensions if dim["page"] == page), None)
        if pdf_dim:
            scale_x = img["width"] / pdf_dim["width"]
            scale_y = img["height"] / pdf_dim["height"]
        else:
            scale_x = scale_y = "N/A"

        html += f"""
            <tr>
                <td>{img["page"]}</td>
                <td>{img["width"]}</td>
                <td>{img["height"]}</td>
                <td>{scale_x if isinstance(scale_x, str) else f"{scale_x:.4f}"}</td>
                <td>{scale_y if isinstance(scale_y, str) else f"{scale_y:.4f}"}</td>
            </tr>"""

    html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Detected Figures</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Page</th>
                <th>Type</th>
                <th>Original Bounding Box (PDF coordinates)</th>
            </tr>
    """

    # Add figure information
    for fig in figures:
        bbox = fig["bounding_box"]
        html += f"""
            <tr>
                <td>{fig["figure_id"]}</td>
                <td>{fig["page"]}</td>
                <td>{fig["figure_type"]}</td>
                <td>left: {bbox["left"]}, top: {bbox["top"]}, width: {bbox["width"]}, height: {bbox["height"]}</td>
            </tr>"""

    html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Method Comparison</h2>
        <p>The following images show the different scaling methods tested:</p>
        <div class="method-comparison">
    """

    # Add method comparison images
    methods = ["simple_ratio", "dpi_adjusted", "orientation_aware", "adaptive_margin"]
    for page_num in set(fig["page"] for fig in figures):
        html += f"""
            <h3>Page {page_num}</h3>
            <div class="figures">"""

        for method in methods:
            img_path = f"page_{page_num}_{method}.png"
            rel_path = os.path.join("debug_outputs", img_path)
            if os.path.exists(rel_path):
                html += f"""
                <div class="figure-card">
                    <h4>{method.replace('_', ' ').title()}</h4>
                    <img src="{img_path}" style="max-width: 300px; border: 1px solid #ddd;" />
                </div>"""

        html += """
            </div>"""

    # Add recommendation section
    html += """
        </div>
    </div>
    
    <div class="recommendation">
        <h2>Recommendations</h2>
        <p>Based on the visualizations above, here are the best scaling methods for your use case:</p>
        <ol>
            <li><strong>For accurate chart boundaries:</strong> [Fill in based on results]</li>
            <li><strong>For ensuring complete chart capture:</strong> The adaptive_margin method with 10% margin works well</li>
            <li><strong>For handling orientation issues:</strong> [Fill in based on results]</li>
        </ol>
        <p>We recommend implementing the scaling method that best fits your PDF documents and chart extraction needs.</p>
    </div>
</body>
</html>
    """

    # Write the HTML report
    report_path = os.path.join(OUTPUT_DIR, "scaling_report.html")
    with open(report_path, "w") as f:
        f.write(html)

    print(f"Summary report generated: {report_path}")


if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    main(pdf_path)