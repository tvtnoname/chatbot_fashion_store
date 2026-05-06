import os

class Config:
    PROJECT_NAME: str = "Fashion Store RAG Chatbot API"
    
    # Base directory paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

settings = Config()
