from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from utils.config_loader import load_config

config = load_config()
CHUNK_SIZE = config['indexing']['chunk_size']
OVERLAP = config['indexing']['overlap']

def chunk_documents(texts):
    """
    문서 리스트를 청크 단위로 분리합니다.
    """
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    docs = [Document(text=t) for t in texts]
    nodes = splitter.get_nodes_from_documents(docs)
    return [n.text for n in nodes]
