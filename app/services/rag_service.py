import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain.tools.retriever import create_retriever_tool
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
        
        policy_tool = create_retriever_tool(
            retriever,
            "policy_retriever",
            "Tìm kiếm và tra cứu các quy định, chính sách đổi trả, vận chuyển, thanh toán của cửa hàng. Bắt buộc phải dùng tool này khi khách hỏi về quy định, chính sách."
        )
        
        # 2. Setup Tools
        tools = [policy_tool, check_inventory, check_order_status, cancel_order]
        
        # 3. LLM Setup
        print("  → Initializing ChatOllama (llama3) with Tool Calling...")
        llm = ChatOllama(model="llama3", temperature=0)
        
        # 4. System Prompt (state_modifier in langgraph)
        system_prompt = (
            "Bạn là 'AI chăm sóc khách hàng' của một cửa hàng thời trang.\n"
            "Luôn trả lời bằng tiếng Việt thân thiện, tự nhiên và lịch sự.\n"
            "Bạn ĐƯỢC CUNG CẤP các công cụ (tools) để tra cứu thông tin (tồn kho, đơn hàng, quy định). Hãy chủ động gọi tool khi cần.\n"
            "Dựa vào kết quả trả về của tool, hãy trả lời ngắn gọn, đúng trọng tâm cho khách hàng.\n"
            "Nếu tool trả về lỗi hoặc không tìm thấy thông tin, hãy thành thật xin lỗi khách hàng và khuyên họ gọi Hotline."
        )
        
        # 5. Agent Executor (langgraph)
        self.agent_executor = create_react_agent(llm, tools=tools, state_modifier=system_prompt)
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
