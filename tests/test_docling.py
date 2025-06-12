from docling.document_converter import DocumentConverter
from pathlib import Path
import os

# Create output directory if it doesn't exist
output_dir = Path("temp_images")
output_dir.mkdir(exist_ok=True)

# Path to the PDF file
pdf_path = "pdf_data/2018 백내장 진료지침.pdf"
page_number = 2  # 0-based index, so page 71 is actually page 72 in the PDF

# Initialize the converter
converter = DocumentConverter()

# Convert the specific page
# Note: docling's convert method accepts a page_range parameter
result = converter.convert(
    pdf_path,
    page_range=(page_number + 1, page_number + 1)  # 1-based page numbers
)

# Print the markdown output
print("\nMarkdown output:")
markdown_output = result.document.export_to_markdown()
print(markdown_output)

# Save to file
output_file = output_dir / f"page_{page_number}_docling_output.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(markdown_output)
print(f"\nSaved markdown output to {output_file}") 