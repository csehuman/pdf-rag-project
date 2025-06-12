from dotenv import load_dotenv
import os
import pickle
import random
from tqdm.auto import tqdm
import secrets
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

from pinecone import Pinecone
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import BM25Encoder
from kiwi import KiwiBM25Tokenizer

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List, Dict, Any, Optional, Tuple

def process_markdown_file(file_path: str) -> Tuple[List[str], Dict[str, List]]:
    """
    Process a single markdown file and return its contents and metadata
    """
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_document = f.read()

    headers_to_split_on = [
        ("#", "heading1"),
        ("##", "heading2"),
        ("###", "heading3"),
        ("####", "heading4"),
        ("#####", "heading5"),
        ("######", "heading6")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    md_header_splits = markdown_splitter.split_text(markdown_document)

    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    split_docs = text_splitter.split_documents(md_header_splits)

    metadata_keys = ["source", "heading1", "heading2", "heading3", "heading4", "heading5", "heading6"]
    metadatas = {key: [] for key in metadata_keys}
    contents = []

    for doc in split_docs:
        doc.metadata.update({"source": file_path})
        contents.append(doc.page_content)
        for k in metadata_keys:
            value = doc.metadata.get(k)
            if value is None:
                metadatas[k].append('')
            else:
                try:
                    metadatas[k].append(int(value))
                except (ValueError, TypeError):
                    metadatas[k].append(value)

    return contents, metadatas

def main():
    # Find all markdown files ending with _merged.md
    markdown_files = glob.glob("processed_md/*_merged.md")
    
    if not markdown_files:
        print("No markdown files found ending with _merged.md")
        return

    print(f"Found {len(markdown_files)} markdown files to process")
    
    # Process all files and combine their contents
    all_contents = []
    all_metadatas = {key: [] for key in ["source", "heading1", "heading2", "heading3", "heading4", "heading5", "heading6"]}
    
    for file_path in tqdm(markdown_files, desc="Processing markdown files"):
        contents, metadatas = process_markdown_file(file_path)
        all_contents.extend(contents)
        for key in all_metadatas:
            all_metadatas[key].extend(metadatas[key])

    # Load environment variables and initialize Pinecone
    load_dotenv()
    api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=api_key, pool_threads=30)

    # Initialize BM25 encoder
    stopword_path = "stopwords-ko.txt"
    with open(stopword_path, "r", encoding="utf-8") as f:
        stopwords_document = f.read()
    stopwords_data = stopwords_document.splitlines()
    stopwords = [word.strip() for word in stopwords_data]

    bm25 = BM25Encoder(language="english")
    bm25._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)

    index_name = "ko-md-strict-multilingual-e5-large-instruct"
    index = pc.Index(index_name)

    save_path = f'./sparse_encoder_{index_name}.pkl'
    bm25.fit(all_contents)
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[fit_sparse_encoder]\nSaved Sparse Encoder to: {save_path}")

    sparse_encoder = bm25

    with open(save_path, "rb") as f:
        sparse_encoder = pickle.load(f)
    print(f"[load_sparse_encoder]\nLoaded Sparse Encoder from: {save_path}")


    def generate_hash() -> str:
        """24자리 무작위 hex 값을 생성하고 6자리씩 나누어 '-'로 연결합니다."""
        random_hex = secrets.token_hex(12)
        return "-".join(random_hex[i : i + 6] for i in range(0, 24, 6))

    keys = list(all_metadatas.keys())
    batch_size = 200
    max_workers = 30

    #model_name = "dragonkue/bge-m3-ko"
    model_name = "intfloat/multilingual-e5-large-instruct"
    embedder = HuggingFaceEmbeddings(
        model_name=model_name
    )

    def chunks(iterable, size):
        it = iter(iterable)
        chunk = list(itertools.islice(it, size))
        while chunk:
            yield chunk
            chunk = list(itertools.islice(it, size))

    def process_batch(batch):
        context_batch = [all_contents[i] for i in batch]
        metadata_batches = {key: [all_metadatas[key][i] for i in batch] for key in keys}

        batch_result = [
            {
                "context": context[:1000],
                **{key: metadata_batches[key][j] for key in keys},
            }
            for j, context in enumerate(context_batch)
        ]

        ids = [generate_hash() for _ in range(len(batch))]
        dense_embeds = embedder.embed_documents(context_batch)
        sparse_embeds = sparse_encoder.encode_documents(context_batch)

        vectors = [
            {
                "id": _id,
                "sparse_values": sparse,
                "values": dense,
                "metadata": metadata,
            }
            for _id, sparse, dense, metadata in zip(
                ids, sparse_embeds, dense_embeds, batch_result
            )
        ]

        try:
            return index.upsert(vectors=vectors, async_req=False)
        except Exception as e:
            print(f"Upsert 중 오류 발생: {e}")
            return None

    batches = list(chunks(range(len(all_contents)), batch_size))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]

        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="문서 Upsert 중"
        ):
            result = future.result()
            if result:
                results.append(result)

    total_upserted = sum(result.upserted_count for result in results if result)
    print(f"총 {total_upserted}개의 Vector 가 Upsert 되었습니다.")
    print(f"{index.describe_index_stats()}")

if __name__ == "__main__":
    main() 