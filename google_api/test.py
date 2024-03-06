from pathlib import Path
from PIL import Image
import google.generativeai as genai

GOOGLE_API_KEY='AIzaSyDMQ-zNh_GzgumWfBVwmUWjNdAaqRdrv5M'
genai.configure(api_key=GOOGLE_API_KEY)

def analyze_image(image_path, prompt="Identify the item in this picture"):
  """Analyzes an image using the Gemini Pro Vision model.

  Args:
      image_path: Path to the image file.
      prompt: A descriptive prompt guiding the model's analysis (optional).

  Returns:
      The generated text describing the image based on the prompt.
  """

  # Configure the API key
  genai.configure(api_key=GOOGLE_API_KEY)

  # Load the image
  try:
      image = Image.open(image_path)
  except FileNotFoundError:
      print(f"Error: Image file not found at {image_path}")
      return None

  # Create the content list
  contents = [prompt, image]

  # Generate response from the model
  model = genai.GenerativeModel("gemini-pro-vision")
  response = model.generate_content(contents, stream=False)

  return response

# Example usage
image_path = Path('images/PXL_20240306_181014408.jpg')
analysis_text = analyze_image(image_path)

if analysis_text:
  print(f"Image analysis: {analysis_text.text}")
else:
  print("Error occurred during image analysis.")