from PIL import Image, ImageDraw

def create_test_image(output_path="test_image.png"):
    """Generates a simple PNG image with colored shapes."""

    # Create a new image with a white background
    width, height = 300, 200
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Draw a red rectangle
    draw.rectangle([(50, 50), (150, 100)], fill="red")

    # Draw a blue circle
    draw.ellipse([(180, 80), (230, 130)], fill="blue")

    # Draw a green triangle
    draw.polygon([(100, 150), (150, 180), (50, 180)], fill="green")

    # Save the image
    image.save(output_path, "PNG")
    print(f"Test image saved to {output_path}")

if __name__ == "__main__":
    create_test_image()