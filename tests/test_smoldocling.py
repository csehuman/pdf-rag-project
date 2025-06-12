# Prerequisites:
# pip install torch
# pip install docling_core
# pip install transformers
# pip install pdf2image
# pip install Pillow

import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from pathlib import Path
from pdf2image import convert_from_path
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f'DEVICE: {DEVICE}')

# Convert PDF page to image
pdf_path = "pdf_data/2018 백내장 진료지침.pdf"
page_number = 3  # 0-based index, so page 71 is actually page 72 in the PDF

# Create output directory if it doesn't exist
output_dir = Path("temp_images")
output_dir.mkdir(exist_ok=True)

# Convert the specific page to an image
images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
if images:
    image = images[0]
    # Save the image temporarily
    image_path = output_dir / f"page_{page_number}.jpg"
    image.save(image_path, "JPEG")
    print(f"Saved page {page_number} to {image_path}")
else:
    raise Exception(f"Could not convert page {page_number} from PDF")

# Load the image
image = load_image(str(image_path))

# Initialize processor and model
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained(
    "ds4sd/SmolDocling-256M-preview",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=8192)
prompt_length = inputs.input_ids.shape[1]
trimmed_generated_ids = generated_ids[:, prompt_length:]
doctags = processor.batch_decode(
    trimmed_generated_ids,
    skip_special_tokens=False,
)[0].lstrip()

# Populate document
doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
print("\nDoctags output:")
print(doctags)

# create a docling document
doc = DoclingDocument(name=f"Page_{page_number}")
doc.load_from_doctags(doctags_doc)

# Save the markdown output
output_md = doc.export_to_markdown()
print("\nMarkdown output:")
print(output_md)

# Save to file
output_file = output_dir / f"page_{page_number}_output.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_md)
print(f"\nSaved markdown output to {output_file}")

# Clean up temporary image
# os.remove(image_path)
# print(f"Cleaned up temporary image: {image_path}")
