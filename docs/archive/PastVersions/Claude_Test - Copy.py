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
    NOTE: Claude is not actually 'seeing' the image; it only receives the base64
          as text and can guess/hallucinate about the image.
    """
    try:
        # Create an Anthropic client
        client = anthropic.Client(api_key=api_key)

        # Read and encode the image to base64
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_encoded_image = base64.b64encode(image_data).decode("utf-8")

        # Prepare the prompt
        prompt_text = f"""
        Please describe what is in this image.

        Here is the image (base64 data):
        {base64_encoded_image}
        """

        # Combine prompt with Anthropic’s HUMAN_PROMPT and AI_PROMPT
        full_prompt = f"{anthropic.HUMAN_PROMPT} {prompt_text} {anthropic.AI_PROMPT}"

        # Call the Claude API
        response = client.completion(
            prompt=full_prompt,
            model="claude-instant-v1",  # or "claude-v1" if you have access
            max_tokens_to_sample=512,
            temperature=0.2,
        )

        # Read the model's text
        completion_text = response.get("completion", "")

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
    # image_path = "test_image.png"
    image_path = r"/.idea/test_image.png"

    if test_claude_api_with_image(api_key, image_path):
        print("Claude image processing test passed.")
    else:
        print("Claude image processing test failed.")
