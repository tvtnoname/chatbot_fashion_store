import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

class RAGService:
    def __init__(self):
        self.retriever = None
        self.llm = None
        self.prompt = None

    def initialize(self):
        print("🔄 Loading RAG components...")
        
        # 1. Embedding model
        print("  → Loading HuggingFace Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        
        # 2. ChromaDB vector store
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")
        print(f"  → Loading ChromaDB from {chroma_path}...")
        vector_store = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 3. Ollama LLM
        print("  → Initializing Ollama LLM (llama3)...")
        self.llm = Ollama(model="llama3")
        
        # 4. Prompt template
        prompt_template = """Bạn là trợ lý chăm sóc khách hàng của cửa hàng thời trang.
Dưới đây là các trích đoạn chính sách từ cửa hàng:
{context}

Dựa VÀO CHÍNH XÁC những thông tin trên (không tự bịa thêm thông tin), hãy trả lời câu hỏi của khách hàng bằng tiếng Việt một cách lịch sự, tự nhiên.
Nếu trong thông tin trên không có câu trả lời, hãy nói rằng "Dạ em chưa có thông tin về vấn đề này. Mong quý khách liên hệ trực tiếp hotline để được hỗ trợ ạ."

Câu hỏi của khách: {question}

Trả lời:"""
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        print("✅ RAG components initialized!")

    def get_answer(self, question: str) -> str:
        if not self.retriever or not self.llm:
            raise RuntimeError("RAG components are not initialized.")
            
        # 1. Retrieve relevant policy chunks
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Build final prompt
        final_prompt = self.prompt.format(context=context, question=question)
        
        # 3. Generate response
        response_text = self.llm.invoke(final_prompt)
        return response_text

# Singleton instance
rag_service = RAGService()
