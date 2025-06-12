import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_upstage import UpstageDocumentParseLoader
from pypdf import PdfReader, PdfWriter

# Load environment variables from .env file
load_dotenv()

# Get API key from .env
os.environ["UPSTAGE_API_KEY"] = os.getenv('UPSTAGE_API_KEY')

# Create output directory if it doesn't exist
output_dir = Path("temp_images")
output_dir.mkdir(exist_ok=True)

# Path to the PDF file
pdf_path = "pdf_data/2010 대한폐암학회.pdf"
page_number = 71  # 0-based index, so page 71 is actually page 72 in the PDF

# Create a temporary PDF with just page 72
temp_pdf_path = output_dir / "temp_page_71.pdf"
reader = PdfReader(pdf_path)
writer = PdfWriter()

# Add only page 72 to the new PDF
writer.add_page(reader.pages[page_number])

# Save the temporary PDF
with open(temp_pdf_path, "wb") as output_file:
    writer.write(output_file)

# Initialize the loader with the temporary PDF
loader = UpstageDocumentParseLoader(str(temp_pdf_path), split="page")

# Load the document (it will only have one page)
docs = loader.load()

# Print the content
print("\nUpstage output:")
print(docs[0])

# Save to file
output_file = output_dir / f"page_{page_number}_upstage_output.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(str(docs[0]))
print(f"\nSaved output to {output_file}")

# Clean up the temporary PDF
temp_pdf_path.unlink() 