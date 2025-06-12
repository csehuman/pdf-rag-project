from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import io
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Initialize Gemini client
api_key = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=api_key)

def convert_doc_to_images(path, page_number):
    """Convert specific page of PDF to image"""
    images = convert_from_path(path, first_page=page_number, last_page=page_number)
    return images[0] if images else None

def get_image_bytes(img):
    """Convert PIL image to bytes"""
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return img_buffer.getvalue()

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("temp_images")
    output_dir.mkdir(exist_ok=True)

    # Path to the PDF file
    pdf_path = "pdf_data/2018 백내장 진료지침.pdf"
    page_number = 1  # 0-based index, so page 71 is actually page 72 in the PDF

    try:
        # Convert the specific page to image
        image = convert_doc_to_images(pdf_path, page_number + 1)  # +1 because pdf2image uses 1-based indexing
        if not image:
            print(f"Failed to convert page {page_number} to image")
            return

        # Save the image temporarily
        temp_img_path = output_dir / f"temp_pg_{page_number + 1}.png"
        image.save(temp_img_path, "PNG")
        print(f"Saved temporary image to {temp_img_path}")

        # Get image bytes
        image_bytes = get_image_bytes(image)

        # The prompt
        prompt = """
Parse the entire textual and structured content of the page into Markdown format. Do not surround your output with triple backticks.
Follow these instructions strictly:
- **Language Preservation:** **ABSOLUTELY PRESERVE THE ORIGINAL LANGUAGE OF EACH TEXT ELEMENT EXACTLY AS IT APPEARS IN THE IMAGE.** Do not translate any text; use the specific language (Korean, English, or others) that you find for each word or phrase.
- **Headings:** Identify headings and subheadings and represent them using appropriate Markdown heading levels (`#`, `##`, `###`, etc.) using the original language text found in the image.
- **Paragraphs:** Represent continuous text as Markdown paragraphs using the original language text found in the image.
- **Lists:** If there are bullet points or numbered lists, use Markdown list syntax (`-` or `1.`) using the original language text found in the image for list items.
- **Tables:** Identify tables and convert them into standard Markdown table format, including headers and rows, using the original language text found in the image for headers and cell content.
- **Figures/Diagrams (like flowcharts):**
    - Identify the figure/diagram and include its title (in its original language from the image).
    - Provide a *description* of the figure/diagram's content and *flow*. When referring to elements within the diagram (like box labels or text), use their exact original language text as extracted from the image.
    - For flowcharts, trace the steps and connections (arrows).
    - For description and steps, use Korean.
- **Special Elements:** Identify any other significant elements (like captions, page footers) and include them using their original language text found in the image.
- **Maintain Order:** Present the Markdown content in the same order as it appears on the page, from top to bottom.
- If something that looks like a page number appear, mark it with <PAGE>{page_no}. ex) <PAGE>1, <PAGE>32 etc. 
"""

        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=1,
            ),
        )

        # Print the response
        print("\nGemini output:")
        print(response.text)

        # Save to file
        output_file = output_dir / f"page_{page_number}_gemini_img_output.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"\nSaved output to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 