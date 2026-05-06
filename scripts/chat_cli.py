import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def chat():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_path = os.path.join(base_dir, "data", "chroma_db")
    
    print("Loading vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    print("Initializing Ollama LLM (llama3)...")
    llm = Ollama(model="llama3")
    
    prompt_template = """Bạn là trợ lý chăm sóc khách hàng của cửa hàng thời trang.
Dưới đây là các trích đoạn chính sách từ cửa hàng:
{context}

Dựa VÀO CHÍNH XÁC những thông tin trên (không tự bịa thêm thông tin), hãy trả lời câu hỏi của khách hàng bằng tiếng Việt một cách lịch sự, tự nhiên.
Nếu trong thông tin trên không có câu trả lời, hãy nói rằng "Dạ em chưa có thông tin về vấn đề này. Mong quý khách liên hệ trực tiếp hotline để được hỗ trợ ạ."

Câu hỏi của khách: {question}

Trả lời:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    print("Chatbot is ready! (Type 'quit' or 'exit' to stop)")
    while True:
        try:
            question = input("\nBạn: ")
            if question.lower() in ['quit', 'exit']:
                print("Tạm biệt!")
                break
                
            if not question.strip():
                continue
                
            # Retrieve relevant chunks
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate the prompt
            final_prompt = prompt.format(context=context, question=question)
            
            # Stream the response
            print("Bot: ", end="", flush=True)
            for chunk in llm.stream(final_prompt):
                print(chunk, end="", flush=True)
            print()
                
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    chat()
