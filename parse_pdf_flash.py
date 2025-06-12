from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import io
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PyPDF2 import PdfReader
import sys
import traceback
from datetime import datetime

# Load environment variables
load_dotenv()

def setup_directories(pdf_name):
    """Create necessary directories for output files"""
    # Create pdf2md directory if it doesn't exist
    pdf2md_dir = Path("pdf2md_flash_v2")
    pdf2md_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for individual page files
    page_dir = pdf2md_dir / pdf_name
    page_dir.mkdir(exist_ok=True)
    
    return pdf2md_dir, page_dir

def log_error(pdf_name, error_message):
    """Log error message to a text file"""
    error_file = Path("pdf2md_flash_v2") / f"{pdf_name}_errors.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {error_message}\n")

def log_progress(pdf_name, message):
    """Log progress message to a text file"""
    progress_file = Path("pdf2md_flash_v2") / f"{pdf_name}_progress.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

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

def process_pdf_page(pdf_path, page_number, client, previous_markdown=""):
    """Process a single page of PDF using Gemini"""

    prompt = f"""
    Parse the entire textual and structured content of the page into Markdown format. Do not surround your output with triple backticks.
    Follow these instructions strictly:
    - **Language Preservation:** **ABSOLUTELY PRESERVE THE ORIGINAL LANGUAGE OF EACH TEXT ELEMENT EXACTLY AS IT APPEARS IN THE IMAGE.** Do not translate any text; use the specific language (Korean, English, or others) that you find for each word or phrase.
    - **Headings:** Identify headings and subheadings and represent them using appropriate Markdown heading levels (`#`, `##`, `###`, etc.) using the original language text found in the image.
    - **Paragraphs:** Represent continuous text as Markdown paragraphs using the original language text found in the image. For the body text, reconstruct full sentences by ignoring line breaks that are solely due to layout and not intended as paragraph breaks.
    - **Lists:** If there are bullet points or numbered lists, use Markdown list syntax (`-` or `1.`) using the original language text found in the image for list items.
    - **Tables:** Identify tables and convert them into standard Markdown table format, including headers and rows, using the original language text found in the image for headers and cell content.
    - **Figures/Diagrams (like flowcharts):**
    - Identify the figure/diagram and include its title (in its original language from the image).
    - Provide a *description* of the figure/diagram's content and *flow*. When referring to elements within the diagram (like box labels or text), use their exact original language text as extracted from the image.
    - For flowcharts, trace the steps and connections (arrows).
    - For description and steps, use Korean.
    - **Special Elements:** Identify any other significant elements (like captions, page footers) and include them using their original language text found in the image.
    - **Maintain Order:** Present the Markdown content in the same order as it appears on the page, from top to bottom.
    - If something that looks like a page number of the content appear in the bottom, mark it with <PAGE>[page_no]. ex) <PAGE>1, <PAGE>32 etc.
    - Maintain and **continue the structural hierarchy and numbering** from the end of the previous page's markdown. If a list item or section number on this page logically follows a list or section on the previous page, continue that numbering/listing at the correct level and sequence, if not just let it be.
    - No need to describe logo-like images.
    ---PREVIOUS_PAGE_MARKDOWN_START---
    {previous_markdown}
    ---PREVIOUS_PAGE_MARKDOWN_END---

    Current Page Image to process:
    """

    try:
        # Convert the specific page to image
        image = convert_doc_to_images(pdf_path, page_number + 1)  # +1 because pdf2image uses 1-based indexing
        if not image:
            print(f"Failed to convert page {page_number} to image")
            return None

        # Get image bytes
        image_bytes = get_image_bytes(image)

        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
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

        return response.text

    except Exception as e:
        print(f"Error processing page {page_number + 1}: {str(e)}")
        return None

def process_single_pdf(pdf_path, client):
    """Process a single PDF file"""
    # Get PDF name without extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    log_progress(pdf_name, f"Starting to process PDF: {pdf_name}")
    print(f"\nProcessing PDF: {pdf_name}")
    
    # Setup directories
    pdf2md_dir, page_dir = setup_directories(pdf_name)
    
    # Get total number of pages
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    log_progress(pdf_name, f"Total pages: {total_pages}")
    print(f"Total pages: {total_pages}")
    
    # Initialize merged content
    merged_content = []
    previous_markdown = ""
    
    # Process each page
    for page_number in range(total_pages):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                log_progress(pdf_name, f"Processing page {page_number + 1} of {total_pages}")
                print(f"Processing page {page_number + 1} of {total_pages}")
                
                # Process the page
                content = process_pdf_page(pdf_path, page_number, client, previous_markdown)
                
                if content is None:
                    error_message = f"{page_number + 1}: None!!!"
                    print(error_message)
                    log_error(pdf_name, error_message)
                    merged_content.append(f"\n")
                    break
                else:
                    # Save individual page content
                    page_file = page_dir / f"{pdf_name}_{page_number + 1}.md"
                    with open(page_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    # Update previous markdown for next iteration
                    previous_markdown = content

                    # Add to merged content
                    merged_content.append(f"\n{content}")
                    log_progress(pdf_name, f"Successfully processed page {page_number + 1}")
                    break
                    
            except Exception as e:
                retry_count += 1
                error_message = f"Error on page {page_number + 1} (Attempt {retry_count}/{max_retries}): {str(e)}"
                print(error_message)
                log_error(pdf_name, error_message)
                
                if retry_count < max_retries:
                    wait_time = 5 * retry_count  # Exponential backoff
                    log_progress(pdf_name, f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    log_error(pdf_name, f"Failed to process page {page_number + 1} after {max_retries} attempts")
                    merged_content.append(f"\n")
    
    # Save merged content
    try:
        merged_file = pdf2md_dir / f"{pdf_name}_merged.md"
        with open(merged_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(merged_content))
        log_progress(pdf_name, f"Completed processing {pdf_name}")
    except Exception as e:
        error_message = f"Error saving merged content: {str(e)}"
        print(error_message)
        log_error(pdf_name, error_message)

def main():
    try:
        # Initialize the Gemini client
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        client = genai.Client(api_key=api_key)

        # Path to the PDF data directory
        pdf_data_dir = Path("pdf_data")
        if not pdf_data_dir.exists():
            raise FileNotFoundError(f"PDF data directory not found at {pdf_data_dir}")

        # Get all PDF files in the directory
        pdf_files = list(pdf_data_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {pdf_data_dir}")

        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF file
        for pdf_path in pdf_files:
            try:
                process_single_pdf(pdf_path, client)
            except Exception as e:
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                error_message = f"Fatal error processing PDF: {str(e)}\n{traceback.format_exc()}"
                print(error_message)
                log_error(pdf_name, error_message)
                continue
                
    except Exception as e:
        error_message = f"Critical error in main: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        log_error("system", error_message)
        sys.exit(1)

if __name__ == "__main__":
    main() 