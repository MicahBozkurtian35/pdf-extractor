import anthropic
import base64
import os

def read_anthropic_key(file_path=r"C:\API\anthropic_key.txt"):
    """Reads the Anthropic API key from a text file and returns it."""
    try:
        with open(file_path, "r") as f:
            api_key = f.read().strip()
            return api_key
    except Exception as e:
        print(f"Error reading Anthropic API key from {file_path}: {e}")
        return None

def test_claude_api_with_image(api_key, image_path):
    """
    Tests Anthropic Claude with a simple prompt containing base64 image data.
    NOTE: Claude does not 'see' the image; it only receives the base64 as text.
    """
    try:
        # Create an Anthropic client
        client = anthropic.Anthropic(api_key=api_key)

        # Read and encode the image to base64
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_encoded_image = base64.b64encode(image_data).decode("utf-8")
        # print(base64_encoded_image)

        # Prepare the prompt
        prompt_text = f"""
        Please describe what is in this image.

        Here is the image (base64 data):
        {base64_encoded_image}
        """

        # Call Claude API (Correct method)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",  # Use the latest Claude 3 model
            max_tokens=512,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe what is in this image."},
                        {"type": "image", "source": {"type": "base64", "data": base64_encoded_image}}
                    ]
                }
            ]
        )


        # Read the model's response
        completion_text = response.content[0].text if response.content else ""

        if completion_text.strip():
            print("Successfully processed image:")
            print(completion_text)
            return True
        else:
            print("Claude returned an empty response.")
            return False

    except Exception as e:
        print("Error with Claude API:", e)
        return False


if __name__ == "__main__":
    # Use a text file containing only the key
    key_file_path = r"C:\API\anthropic_key.txt"
    api_key = read_anthropic_key(key_file_path)

    if not api_key:
        print("Could not read Anthropic API key, exiting.")
        exit(1)

    # Path to a test image
    image_path = r"/.idea/test_image.png"

    if test_claude_api_with_image(api_key, image_path):
        print("Claude image processing test passed.")
    else:
        print("Claude image processing test failed.")
