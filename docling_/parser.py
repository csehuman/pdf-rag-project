import os
# import io
import glob
import fitz  # PyMuPDF
import tempfile
import time
# import pytesseract
# import cv2
# import numpy as np
# from PIL import Image
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# 디버깅 이미지 저장 폴더
DEBUG_IMAGE_FOLDER = "debug_pages"
os.makedirs(DEBUG_IMAGE_FOLDER, exist_ok=True)

# Docling 설정
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions()
        )
    }
)

# def apply_ocr_to_pdf_page(pdf_path: str, page_number: int) -> str:
#     # PDF → 이미지
#     doc = fitz.open(pdf_path)
#     page = doc.load_page(page_number)
#     pix = page.get_pixmap(dpi=300)
#     img_bytes = pix.tobytes()
 
#     # 디버그 이미지 저장
#     os.makedirs(DEBUG_IMAGE_FOLDER, exist_ok=True)
#     file_name = os.path.basename(pdf_path).replace(".pdf", "")
#     save_path = os.path.join(DEBUG_IMAGE_FOLDER, f"{file_name}_page_{page_number + 1}.png")
#     with open(save_path, "wb") as f:
#         f.write(img_bytes)

#     # 이미지 불러오기
#     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     img_np = np.array(img)

#     # 🔧 전처리 1: Grayscale
#     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

#     # 🔧 전처리 2: Adaptive Thresholding + MedianBlur
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, 
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY, 
#         11, 2
#     )
#     processed_img = cv2.medianBlur(thresh, 3)

#     # PIL 변환
#     preprocessed_pil = Image.fromarray(processed_img)

#     # 🔍 OCR with custom config
#     custom_config = r'--oem 3 --psm 6'  # PSM 6: Uniform block of text
#     text = pytesseract.image_to_string(preprocessed_pil, lang="kor+eng", config=custom_config)

#     # 🧹 클린업
#     md_text = text.strip().replace('\f', '').replace('\r\n', '\n')

#     return md_text

# 텍스트 품질 검사 기준
def is_extraction_failed(text: str, min_length=30) -> bool:
    return len(text.strip()) < min_length or "표" in text[:20]

def extract_text_from_pdf(pdf_path: str) -> str:
    result_text = []
    doc = fitz.open(pdf_path)
    num_pages = len(doc)

    for i in range(num_pages):
        page_header = f"## Page {i + 1}"
        try:
            # 메모리에서 한 페이지짜리 PDF 생성 → 임시 파일로 저장
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=i, to_page=i)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                single_page_doc.save(temp_pdf.name)
                temp_pdf_path = Path(temp_pdf.name)

            # Docling 변환 시도
            try:
                result = converter.convert(temp_pdf_path)
                md = result.document.export_to_markdown()
                text = "\n".join(md) if isinstance(md, list) else md

                if is_extraction_failed(text):
                    raise ValueError("텍스트 품질 기준 미달 → OCR로 대체")

            except Exception as inner_e:
                print(f"[DOC→MARKDOWN 실패] {i+1}쪽 Docling 변환 불가 → OCR 대체: {inner_e}")
                # text = apply_ocr_to_pdf_page(pdf_path, i)
                page_header += " [OCR]"

            result_text.append(f"{page_header}\n\n{text}")
            os.remove(temp_pdf_path)

        except Exception as e:
            print(f"[ERROR] {i+1}쪽 처리 실패 → OCR 적용: {e}")
            # text = apply_ocr_to_pdf_page(pdf_path, i)
            # result_text.append(f"{page_header} [OCR]\n\n{text}")

    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{filename}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(result_text))

    return "\n\n".join(result_text)

# 🔄 PDF 폴더 내 모든 파일을 처리
def load_all_pdfs(folder_path: str) -> list[str]:
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    processed_files = []
    for path in pdf_paths:
        start_time = time.time()
        processed_files.append(extract_text_from_pdf(path))
        finish_time = time.time()
        print("소요 시간 : " + str(finish_time - start_time))
    return processed_files
