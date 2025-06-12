from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

file_path = "pdf2md_flash_v2/2010 대한폐암학회_merged.md"
output_path_1 = "pdf2md_flash_v2/2010 대한폐암학회_merged_split_v2.md"
output_path_2 = "pdf2md_flash_v2/2010 대한폐암학회_merged_split_more.md"

with open(file_path, "r", encoding="utf-8") as f:
    markdown_document = f.read()

# print(markdown_document[50])

headers_to_split_on = [
    ("#", "Heading1"),
    ("##", "Heading2"),
    ("###", "Heading3"),
    ("####", "Heading4"),
    ("#####", "Heading5"),
    ("######", "Heading6")
]


markdown_splitter = MarkdownHeaderTextSplitter(
    # 분할할 헤더를 지정합니다.
    headers_to_split_on=headers_to_split_on,
    # 헤더를 제거하지 않도록 설정합니다.
    strip_headers=False,
)

# 마크다운 문서를 헤더를 기준으로 분할합니다.
md_header_splits = markdown_splitter.split_text(markdown_document)

# 분할된 결과를 출력합니다.
# for header in md_header_splits:
#     print(f"{header.page_content}")
#     print(f"{header.metadata}", end="\n=====================\n")
# Write the split content to a new file
# with open(output_path_1, "w", encoding="utf-8") as out_file:
#     for idx, header in enumerate(md_header_splits):
#         out_file.write(f"<!-- Split Section {idx+1} -->\n")
#         out_file.write(header.page_content.strip() + "\n\n")
#         out_file.write(f"<!-- Metadata: {header.metadata} -->\n")
#         out_file.write("=" * 40 + "\n\n")

# print(f"Split content saved to: {output_path_1}")

chunk_size = 500  # 분할된 청크의 크기를 지정합니다.
chunk_overlap = 50  # 분할된 청크 간의 중복되는 문자 수를 지정합니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# 문서를 문자 단위로 분할합니다.
splits = text_splitter.split_documents(md_header_splits)
# 분할된 결과를 출력합니다.
with open(output_path_2, "w", encoding="utf-8") as out_file:
    for idx, header in enumerate(splits):
        out_file.write(f"<!-- Split Section {idx+1} -->\n")
        out_file.write(header.page_content + "\n\n")
        out_file.write(f"<!-- Metadata: {header.metadata} -->\n")
        out_file.write("=" * 40 + "\n\n")

print(f"Split content saved to: {output_path_2}")