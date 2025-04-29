import fitz  # PyMuPDF
import os
import glob

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def load_all_pdfs(folder_path: str) -> list[str]:
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    return [extract_text_from_pdf(path) for path in pdf_paths]
