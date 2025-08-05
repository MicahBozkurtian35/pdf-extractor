import fitz  # PyMuPDF
import os
import json
import re
import numpy as np
from PIL import Image
import io

def debug_print(*args, **kwargs):
    print(*args, **kwargs)

def extract_figures_from_upstage_response(html_content):
    """Extract figure bounding boxes from Upstage HTML response."""
    figures = []

    # Extract page dimensions from HTML if available
    page_dims_pattern = r'<div class="page" data-width="(\d+)" data-height="(\d+)"'
    page_dims_match = re.search(page_dims_pattern, html_content)
    page_width = int(page_dims_match.group(1)) if page_dims_match else None
    page_height = int(page_dims_match.group(2)) if page_dims_match else None

    if page_width and page_height:
        debug_print(f"Page dimensions from HTML: {page_width} x {page_height}")

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

        figure = {
            "figure_id": fig_id,
            "page": 1,  # Default to page 1
            "bounding_box": {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "right": right,
                "bottom": bottom
            },
            "figure_type": fig_type,
            "original_order": i
        }

        figures.append(figure)
        debug_print(f"Figure {i+1} - Upstage coordinates: left={left}, top={top}, right={right}, bottom={bottom}, width={width}, height={height}")

    return figures, page_width, page_height

def analyze_whitespace(page, rect, direction="up", max_distance=100, threshold=0.95):
    """
    Analyze the whitespace in a direction to find where content begins.

    Args:
        page: PyMuPDF page object
        rect: The initial rectangle to expand from
        direction: "up", "down", "left", or "right"
        max_distance: Maximum distance to search
        threshold: Whitespace threshold (higher = more sensitive)

    Returns:
        Optimal distance to extend in the specified direction
    """
    # Determine scan parameters based on direction
    if direction == "up":
        x_start, x_end = rect.x0, rect.x1
        y_start, y_end = max(0, rect.y0 - max_distance), rect.y0
        step = 1  # Pixels per step
        dimension = "vertical"
    elif direction == "down":
        x_start, x_end = rect.x0, rect.x1
        y_start, y_end = rect.y1, min(page.rect.height, rect.y1 + max_distance)
        step = 1
        dimension = "vertical"
    elif direction == "left":
        x_start, x_end = max(0, rect.x0 - max_distance), rect.x0
        y_start, y_end = rect.y0, rect.y1
        step = 1
        dimension = "horizontal"
    else:  # right
        x_start, x_end = rect.x1, min(page.rect.width, rect.x1 + max_distance)
        y_start, y_end = rect.y0, rect.y1
        step = 1
        dimension = "horizontal"

    # Create test rectangles
    if dimension == "vertical":
        width = rect.width
        extension_distance = int(abs(y_end - y_start))  # Convert to int for range

        # Adjust for scan direction
        if direction == "up":
            test_rect = fitz.Rect(x_start, y_end - step, x_end, y_end)
            scan_direction = -1  # Moving upward
        else:  # down
            test_rect = fitz.Rect(x_start, y_start, x_end, y_start + step)
            scan_direction = 1  # Moving downward

    else:  # horizontal
        height = rect.height
        extension_distance = int(abs(x_end - x_start))  # Convert to int for range

        # Adjust for scan direction
        if direction == "left":
            test_rect = fitz.Rect(x_end - step, y_start, x_end, y_end)
            scan_direction = -1  # Moving leftward
        else:  # right
            test_rect = fitz.Rect(x_start, y_start, x_start + step, y_end)
            scan_direction = 1  # Moving rightward

    # Initialize optimal extension
    optimal_extension = 0
    last_content_pos = None

    # Scan in small steps
    for i in range(0, extension_distance, step):
        if dimension == "vertical":
            # Update test rectangle position
            test_y = y_start if direction == "down" else (y_end - i - step)
            test_rect = fitz.Rect(x_start, test_y, x_end, test_y + step)
        else:  # horizontal
            # Update test rectangle position
            test_x = x_start if direction == "right" else (x_end - i - step)
            test_rect = fitz.Rect(test_x, y_start, test_x + step, y_end)

        # Get pixmap of the small area
        pix = page.get_pixmap(clip=test_rect, matrix=fitz.Matrix(1, 1))

        # Convert to PIL Image for analysis
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # Convert to grayscale and analyze
        img_gray = img.convert('L')
        img_data = np.array(img_gray)

        # Check if this area is mostly white (or very light)
        whiteness = np.mean(img_data) / 255.0  # Normalize to 0-1

        # If we find content (non-white area)
        if whiteness < threshold:
            if last_content_pos is None:
                last_content_pos = i
            # We extend 10px past the last content to ensure we capture everything
            optimal_extension = i + 10

    # Ensure we have a reasonable extension
    if optimal_extension == 0 and last_content_pos is not None:
        optimal_extension = last_content_pos + 10

    # Cap at max_distance
    optimal_extension = min(optimal_extension, max_distance)

    debug_print(f"Optimal {direction} extension: {optimal_extension} pixels")
    return optimal_extension

def simpler_extension(page, figure_rect, top_extend=75, bottom_extend=40, padding=10):
    """
    A simpler approach to extend figure rectangles by percentage of figure height.

    Args:
        page: PyMuPDF page object
        figure_rect: Original figure rectangle
        top_extend: Pixels to extend upward
        bottom_extend: Pixels to extend downward
        padding: Extra padding to add horizontally

    Returns:
        Extended rectangle
    """
    # Apply extensions and padding
    extended_rect = fitz.Rect(
        figure_rect.x0 - padding,  # left
        max(0, figure_rect.y0 - top_extend),  # top (don't go below 0)
        figure_rect.x1 + padding,  # right
        min(page.rect.height, figure_rect.y1 + bottom_extend)  # bottom (don't exceed page height)
    )

    # Ensure the rectangle is within page boundaries
    page_rect = page.rect
    extended_rect.intersect(page_rect)

    debug_print(f"Simple extension: Top={top_extend}px, Bottom={bottom_extend}px")
    debug_print(f"Original: {figure_rect}")
    debug_print(f"Extended: {extended_rect}")

    return extended_rect

def visualize_extension(pdf_path, figures, output_dir="figure_extension"):
    """Visualize figure extension on the PDF."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)

        # Process each page with figures
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Get PDF page dimensions
            pdf_width = page.rect.width
            pdf_height = page.rect.height
            debug_print(f"PDF page dimensions: {pdf_width} x {pdf_height}")

            # Find figures on this page
            page_figures = [f for f in figures if f.get("page", 1) == page_num]

            if not page_figures:
                continue

            # Create a visualization PDF
            vis_doc = fitz.open()
            vis_page = vis_doc.new_page(width=pdf_width, height=pdf_height)

            # Copy content from original page
            vis_page.show_pdf_page(
                vis_page.rect,
                doc,
                page_idx
            )

            # Add header and information
            vis_page.insert_text((20, 20), "Figure Extension Visualization", fontsize=12, color=(0, 0, 0))
            vis_page.insert_text((20, 40), f"PDF dimensions: {pdf_width:.1f} x {pdf_height:.1f}", fontsize=10, color=(0, 0, 0))
            vis_page.insert_text((20, 60), "Blue: Original figure boundaries", fontsize=10, color=(0, 0, 1))
            vis_page.insert_text((20, 80), "Red: Extended boundaries (75px up, 40px down)", fontsize=10, color=(1, 0, 0))

            # Process each figure
            for fig_idx, fig in enumerate(page_figures):
                bbox = fig.get("bounding_box")
                if not bbox:
                    continue

                # Get original coordinates
                left = bbox.get("left", 0)
                top = bbox.get("top", 0)
                width = bbox.get("width", 0)
                height = bbox.get("height", 0)
                right = left + width
                bottom = top + height

                # Create original rectangle
                original_rect = fitz.Rect(left, top, right, bottom)

                # Draw original rectangle
                vis_page.draw_rect(original_rect, color=(0, 0, 1), width=1.5)
                vis_page.insert_text(
                    (left + 5, top + 15),
                    f"Fig {fig_idx+1} (Original)",
                    fontsize=8,
                    color=(0, 0, 1)
                )

                # Use simpler extension approach
                extended_rect = simpler_extension(page, original_rect, top_extend=75, bottom_extend=40)

                # Draw extended rectangle
                vis_page.draw_rect(extended_rect, color=(1, 0, 0), width=2)
                vis_page.insert_text(
                    (extended_rect.x0 + 5, extended_rect.y0 + 30),
                    f"Fig {fig_idx+1} (Extended)",
                    fontsize=8,
                    color=(1, 0, 0)
                )

                # Show original and extended dimensions
                original_dims = f"Original: {original_rect.width:.0f}x{original_rect.height:.0f}"
                extended_dims = f"Extended: {extended_rect.width:.0f}x{extended_rect.height:.0f}"
                vis_page.insert_text(
                    (extended_rect.x0 + 5, extended_rect.y0 + 45),
                    original_dims,
                    fontsize=7,
                    color=(0, 0, 1)
                )
                vis_page.insert_text(
                    (extended_rect.x0 + 5, extended_rect.y0 + 60),
                    extended_dims,
                    fontsize=7,
                    color=(1, 0, 0)
                )

                debug_print(f"Figure {fig_idx+1}:")
                debug_print(f"  Original: {original_rect}")
                debug_print(f"  Extended: {extended_rect}")
                debug_print(f"  Top extension: {original_rect.y0 - extended_rect.y0:.1f}px")
                debug_print(f"  Bottom extension: {extended_rect.y1 - original_rect.y1:.1f}px")

                # Extract both versions for comparison
                try:
                    # Original extraction
                    pix_original = page.get_pixmap(clip=original_rect, matrix=fitz.Matrix(1.5, 1.5))
                    original_path = os.path.join(output_dir, f"original_extract_page{page_num}_fig{fig_idx+1}.png")
                    pix_original.save(original_path)

                    # Extended extraction
                    pix_extended = page.get_pixmap(clip=extended_rect, matrix=fitz.Matrix(1.5, 1.5))
                    extended_path = os.path.join(output_dir, f"extended_extract_page{page_num}_fig{fig_idx+1}.png")
                    pix_extended.save(extended_path)

                    debug_print(f"Saved extractions to {output_dir}")

                except Exception as e:
                    debug_print(f"Error extracting figure {fig_idx+1}: {e}")

            # Save the visualization PDF
            vis_pdf_path = os.path.join(output_dir, f"extension_vis_page_{page_num}.pdf")
            vis_doc.save(vis_pdf_path)
            vis_doc.close()

            # Also save a PNG version for the HTML report
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            png_path = os.path.join(output_dir, f"extension_vis_page_{page_num}.png")
            pix.save(png_path)

            debug_print(f"Saved visualization to {vis_pdf_path}")

            # Create HTML comparison report
            create_html_report(output_dir, page_num, len(page_figures))

        doc.close()
        print(f"Figure extensions visualized in {output_dir}")

    except Exception as e:
        print(f"Error in figure extension: {e}")
        import traceback
        traceback.print_exc()

def create_html_report(output_dir, page_num, num_figures):
    """Create an HTML report comparing original and extended extractions."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Figure Extension Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .comparison { display: flex; margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; }
            .extraction { margin: 10px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            h2 { margin-top: 30px; }
            .original-box { border-left: 5px solid blue; padding-left: 10px; }
            .extended-box { border-left: 5px solid red; padding-left: 10px; }
        </style>
    </head>
    <body>
        <h1>Figure Extension Results</h1>
        <p>This report shows a comparison between the original Upstage coordinates (blue) and 
        the extended coordinates (red) that include titles, labels, and other figure elements.</p>
        
        <div>
            <h2>Full Page Reference</h2>
            <img src="extension_vis_page_""" + str(page_num) + """.png" alt="Visualization PDF" style="max-width: 100%;">
        </div>
    """

    # For each figure
    for fig_idx in range(1, num_figures + 1):
        html += f"""
        <h2>Figure {fig_idx}</h2>
        <div class="comparison">
            <div class="extraction original-box">
                <h3>Original Extraction</h3>
                <img src="original_extract_page{page_num}_fig{fig_idx}.png" alt="Original extraction">
                <p>Using exact Upstage coordinates</p>
            </div>
            
            <div class="extraction extended-box">
                <h3>Extended Extraction</h3>
                <img src="extended_extract_page{page_num}_fig{fig_idx}.png" alt="Extended extraction">
                <p>Boundaries extended to include titles and labels</p>
            </div>
        </div>
        """

    html += """
    </body>
    </html>
    """

    # Write the HTML report
    with open(os.path.join(output_dir, f"extension_report_page_{page_num}.html"), 'w') as f:
        f.write(html)

    debug_print(f"Created HTML report at {output_dir}/extension_report_page_{page_num}.html")

def test_upstage_response(upstage_response_file, pdf_path):
    """Test figure extension with an Upstage response file."""
    try:
        # Load Upstage response from file
        with open(upstage_response_file, 'r') as f:
            upstage_response = json.load(f)

        # Extract HTML content
        if "content" in upstage_response and "html" in upstage_response["content"]:
            html_content = upstage_response["content"]["html"]
            figures, _, _ = extract_figures_from_upstage_response(html_content)

            if figures:
                print(f"Found {len(figures)} figures in Upstage response")
                visualize_extension(pdf_path, figures)
            else:
                print("No figures found in Upstage response")
        else:
            print("No HTML content found in Upstage response")

    except Exception as e:
        print(f"Error processing Upstage response: {e}")
        import traceback
        traceback.print_exc()

def test_manual_coordinates(pdf_path, coordinates_list):
    """Test figure extension with manually specified coordinates."""
    figures = []

    for idx, (page_num, left, top, width, height) in enumerate(coordinates_list):
        figure = {
            "figure_id": f"manual_{idx+1}",
            "page": page_num,
            "bounding_box": {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "right": left + width,
                "bottom": top + height
            },
            "figure_type": "chart",
            "original_order": idx
        }
        figures.append(figure)

    visualize_extension(pdf_path, figures)

# Example usage:
if __name__ == "__main__":
    pdf_path = r"C:\Users\Mbomm\IdeaProjects\PDF Graph Scanner\input_pdfs\Sample3.pdf"

    # These are sample manual coordinates
    sample_coordinates = [
        # Top-left chart
        [1, 45, 145, 240, 140],
        # Top-right chart
        [1, 300, 145, 240, 140],
        # Bottom-left chart
        [1, 45, 385, 240, 140],
        # Bottom-right chart
        [1, 300, 385, 240, 140],
        # Bottom heat map table
        [1, 45, 625, 490, 140]
    ]

    # Test with manual coordinates
    test_manual_coordinates(pdf_path, sample_coordinates)

    print("\nThis script demonstrates a simplified figure extension approach that works reliably across different PDFs.")
    print("Blue boxes: Original figure boundaries")
    print("Red boxes: Extended boundaries")
    print("\nExtension parameters (adjust as needed):")
    print("- Top extension: 75 pixels (captures titles and headers)")
    print("- Bottom extension: 40 pixels (captures source notes and labels)")
    print("- Side padding: 10 pixels on each side")
    print("\nCheck the figure_extension directory for:")
    print("1. PDF/PNG visualization showing both boundary sets")
    print("2. Extracted images comparing original vs. extended results")
    print("3. HTML comparison report")