from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import base64
import io
import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def convert_doc_to_images(path, page_number):
    """Convert specific page of PDF to image"""
    images = convert_from_path(path, first_page=page_number, last_page=page_number)
    return images[0] if images else None

def get_img_uri(img):
    """Convert image to base64 data URI"""
    png_buffer = io.BytesIO()
    img.save(png_buffer, format="PNG")
    png_buffer.seek(0)
    base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_png}"

system_prompt = '''
Extract and structure the content from this PDF document. Include all text, tables, and figures in original language. Format the output strictly in markdown.
'''

def analyze_image(data_uri):
    """Analyze image using GPT-4.1 Vision API"""
    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            },
        ],
        max_tokens=5000,
        temperature=0,
        top_p=0.1
    )
    return response.choices[0].message.content

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("temp_images")
    output_dir.mkdir(exist_ok=True)

    # Path to the PDF file
    pdf_path = "pdf_data/2018 백내장 진료지침.pdf"
    page_number = 2  # 0-based index, so page 71 is actually page 72 in the PDF

    try:
        # Convert the specific page to image
        image = convert_doc_to_images(pdf_path, page_number + 1)  # +1 because pdf2image uses 1-based indexing
        if not image:
            print(f"Failed to convert page {page_number} to image")
            return

        # Convert image to data URI
        data_uri = get_img_uri(image)

        # Analyze the image using GPT-4 Vision
        analysis_result = analyze_image(data_uri)

        # Save to file
        output_file = output_dir / f"page_{page_number}_gpt_vision_output.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(analysis_result)
        print(f"\nSaved GPT-4 Vision analysis to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 