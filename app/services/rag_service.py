import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from app.tools.api_tools import check_inventory, check_order_status, cancel_order

class RAGService:
    def __init__(self):
        self.agent_executor = None

    def initialize(self):
        print("🔄 Loading AI Agent components...")
        
        # 1. Embedding & Vector Store (RAG Tool)
        print("  → Loading HuggingFace Embeddings & ChromaDB...")
        embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")
        vector_store = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        from langchain_core.tools import tool
        
        @tool
        def policy_retriever(query: str) -> str:
            """Tìm kiếm và tra cứu các quy định, chính sách đổi trả, thanh toán của cửa hàng. Bắt buộc phải dùng tool này khi khách hỏi về quy định, chính sách."""
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Setup Tools
        tools = [policy_retriever, check_inventory, check_order_status, cancel_order]
        
        # 3. LLM Setup
        print("  → Initializing ChatOllama (llama3.1) with Tool Calling...")
        llm = ChatOllama(model="llama3.1", temperature=0)
        
        # 4. System Prompt - Strict Tool-Calling Rules
        system_prompt = """Bạn là "AI chăm sóc khách hàng" của cửa hàng thời trang. Luôn trả lời bằng tiếng Việt.

## QUY TẮC BẮT BUỘC - KHÔNG ĐƯỢC VI PHẠM:

**QUY TẮC 0 - Giao tiếp thông thường:**
Nếu khách chỉ chào hỏi hoặc trò chuyện (như "Chào bạn", "Hello", "Hi"):
→ BẠN PHẢI TRẢ LỜI TRỰC TIẾP BẰNG VĂN BẢN (TEXT). 
→ TUYỆT ĐỐI KHÔNG ĐƯỢC GỌI BẤT KỲ TOOL NÀO (KỂ CẢ TOOL ẢO). CẤM TRẢ VỀ JSON CHO TRƯỜNG HỢP NÀY.

**QUY TẮC 1 - Hỏi về đơn hàng:**
Nếu khách hỏi bất cứ điều gì liên quan đến "đơn hàng", "order", "trạng thái", "đơn của tôi", "hủy đơn":
→ BẮT BUỘC gọi tool `check_order_status` với `user_id` lấy từ đầu tin nhắn "[User ID: X]"
→ KHÔNG được tự trả lời mà không dùng tool
→ KHÔNG được bảo khách "vào tài khoản xem"

**QUY TẮC 2 - Hỏi về tồn kho, sản phẩm:**
Nếu khách hỏi về "còn hàng", "tồn kho", "size", "màu", "SKU", "sản phẩm nào":
→ BẮT BUỘC gọi tool `check_inventory` với từ khóa tìm kiếm
→ KHÔNG được tự bịa ra thông tin

**QUY TẮC 3 - Hỏi về chính sách, quy định:**
Nếu khách hỏi về "đổi trả", "bảo hành", "thanh toán", "khuyến mãi", "chính sách":
→ BẮT BUỘC gọi tool `policy_retriever` để tra cứu
→ KHÔNG được tự bịa chính sách

**QUY TẮC 4 - Hủy đơn:**
Nếu khách muốn "hủy đơn", "cancel":
→ Gọi `check_order_status` trước để kiểm tra trạng thái
→ Nếu đơn đang PENDING hoặc CONFIRMED: gọi tiếp `cancel_order`
→ Nếu đơn đang ở trạng thái không cho phép hủy: báo không thể hủy

## LƯU Ý QUAN TRỌNG VỀ CÁCH TRẢ LỜI:
- User ID của khách luôn nằm ở đầu tin nhắn dạng "[User ID: X]" - hãy dùng số X đó. Nếu User ID là "Chưa đăng nhập": nhắc khách đăng nhập.
- **TUYỆT ĐỐI KHÔNG** bao giờ được nói với khách hàng những câu như "Tôi đã gọi tool..." hay "Hệ thống đang dùng công cụ...". Khách hàng không cần biết bạn dùng tool gì.
- Sau khi tool trả về kết quả, BẠN PHẢI trích xuất thông tin đó và trả lời tự nhiên cho khách (Ví dụ: "Dạ, đơn hàng của anh/chị đang ở trạng thái X")."""

        # 5. Agent Executor (langgraph)
        self.agent_executor = create_react_agent(llm, tools=tools, prompt=system_prompt)
        print("✅ AI Agent initialized with LangGraph!")

    def get_answer(self, question: str, user_id: int = None) -> str:
        if not self.agent_executor:
            raise RuntimeError("Agent components are not initialized.")
            
        try:
            # Chuyền user_id thực tế vào context
            uid = user_id if user_id else "Chưa đăng nhập"
            
            # Khởi tạo messages cho langgraph
            inputs = {
                "messages": [
                    ("user", f"[User ID: {uid}] {question}")
                ]
            }
            
            # stream_mode="values" để lấy state cuối cùng
            response_state = self.agent_executor.invoke(inputs)
            
            # Lấy tin nhắn cuối cùng từ mảng messages
            final_message = response_state["messages"][-1].content
            return final_message
        except Exception as e:
            print(f"Agent execution error: {e}")
            import traceback
            traceback.print_exc()
            return "Dạ hệ thống đang quá tải hoặc gặp lỗi. Anh/chị vui lòng thử lại sau nhé."

# Singleton instance
rag_service = RAGService()
