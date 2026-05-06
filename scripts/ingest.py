import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "data", "raw", "quy_dinh_cua_hang_thoi_trang.json")
    chroma_path = os.path.join(base_dir, "data", "chroma_db")
    
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}")
        return

    documents = []
    for item in data:
        # Create a combined text representation for better semantic search
        content = f"Câu hỏi: {item['instruction']}\n{item['input']}\nTrả lời: {item['output']}"
        
        doc = Document(
            page_content=content,
            metadata={"source": "quy_dinh_cua_hang_thoi_trang.json"}
        )
        documents.append(doc)

    print(f"Created {len(documents)} document chunks.")
    
    print("Initializing HuggingFace Embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    print(f"Creating ChromaDB at {chroma_path}...")
    # Clear existing DB if any (Chroma handles this automatically if persist_directory is same, but we are overwriting)
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=chroma_path
    )
    
    print("✅ Ingestion complete. Vector database is ready.")

if __name__ == "__main__":
    main()
