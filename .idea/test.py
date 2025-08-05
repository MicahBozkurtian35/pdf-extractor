import google.generativeai as genai
import base64
import os

def test_gemini_api_with_image(api_key, image_path):
    """Tests Gemini API with a simple image processing request."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro") # Use "gemini-pro" model for image processing

        # Read and encode image to base64
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_encoded_image = base64.b64encode(image_data).decode("utf-8")

        # Simple image processing prompt
        prompt = f"""
        Please describe what is in this image.
        Here is the image: {base64_encoded_image}
        """

        response = model.generate_content(prompt)

        if response.text:
            print("Successfully processed image:")
            print(response.text)
            return True
        else:
            print("Gemini API returned an empty response.")
            return False
    except Exception as e:
        print("Error with Gemini API:", e)
        return False


if __name__ == "__main__":
    # Replace with your actual API Key
    api_key = "AIzaSyBLrI96YkDjaLIHW9Ib4JV4odkDthD-2rU"

    # Path to a test image
    image_path = "test_image.png" # You can use the same test graph image as before

    if test_gemini_api_with_image(api_key, image_path):
        print("Gemini API image processing test passed.")
    else:
        print("Gemini API image processing test failed.")