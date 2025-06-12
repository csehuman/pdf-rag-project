import os
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from pypdf import PdfReader, PdfWriter

# Load environment variables from .env file
load_dotenv()

# Create output directory if it doesn't exist
output_dir = Path("temp_images")
output_dir.mkdir(exist_ok=True)

# Path to the PDF file
pdf_path = "pdf_data/2010 대한폐암학회.pdf"
page_number = 71  # 0-based index, so page 71 is actually page 72 in the PDF

# Create a temporary PDF with just page 71
temp_pdf_path = output_dir / "temp_page_71.pdf"
reader = PdfReader(pdf_path)
writer = PdfWriter()

# Add only page 71 to the new PDF
writer.add_page(reader.pages[page_number])

# Save the temporary PDF
with open(temp_pdf_path, "wb") as output_file:
    writer.write(output_file)

# Initialize LlamaParse
parser = LlamaParse(
    api_key=os.getenv("LLAMACLOUD_API_KEY"),
    result_type="markdown"
)

# Parse the temporary PDF
with open(temp_pdf_path, "rb") as f:
    extra_info = {"file_name": str(temp_pdf_path)}
    documents = parser.load_data(f, extra_info=extra_info)

# Print the content
print("\nLlamaParse output:")
for doc in documents:
    print(doc.text)

# Save to file
output_file = output_dir / f"page_{page_number}_llamaparse_output.md"
with open(output_file, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
print(f"\nSaved output to {output_file}")

# Clean up the temporary PDF
temp_pdf_path.unlink() 