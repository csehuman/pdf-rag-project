from google import genai
from google.genai import types
import pathlib
import os
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
import io

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

# Create output directory if it doesn't exist
output_dir = Path("temp_images")
output_dir.mkdir(exist_ok=True)

# Path to the PDF file
pdf_path = "pdf_data/2018 백내장 진료지침.pdf"
page_number = 2  # 0-based index, so page 71 is actually page 72 in the PDF

# Extract single page from PDF
reader = PdfReader(pdf_path)
writer = PdfWriter()
writer.add_page(reader.pages[page_number])

# Create a temporary PDF file for the single page
temp_pdf_path = output_dir / f"temp_pg_{page_number + 1}.pdf"
with open(temp_pdf_path, "wb") as temp_file:
    writer.write(temp_file)

# Read the temporary PDF file
with open(temp_pdf_path, "rb") as temp_file:
    pdf_bytes = temp_file.read()

# Initialize the client
client = genai.Client(api_key=api_key)

# prompt = "Extract and structure the content from this PDF document. Include all text, tables, and figures in original language. Format the output strictly in markdown."
prompt = """
Parse the entire textual and structured content of the page into Markdown format. Do not sorround your output with triple backticks.
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
"""

# Generate content using Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents=[
        types.Part.from_bytes(
            data=pdf_bytes,
            mime_type='application/pdf',
        ),
        prompt
    ],
    config=types.GenerateContentConfig(
        temperature=0,
    ),
)

# Print the response
print("\nGemini output:")
print(response.text)

# Save to file
output_file = output_dir / f"page_{page_number}_gemini_output.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(response.text)
print(f"\nSaved output to {output_file}")

# Clean up the temporary PDF file
#os.remove(temp_pdf_path)