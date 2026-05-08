"""
RAG Service - Tầng Service chính.
"""
import uuid
import json
from app.agents.graph import build_multi_agent_graph


class RAGService:
    def __init__(self):
        self.agent_graph = None

    def initialize(self):
        """Khởi tạo toàn bộ hệ thống Multi-Agent."""
        print("🔄 Loading Multi-Agent System...")
        self.agent_graph = build_multi_agent_graph()
        print("✅ Multi-Agent System ready!")

    def get_answer(self, question: str, user_id: int = None, thread_id: str = None) -> dict:
        """
        Hàm chính để lấy câu trả lời từ AI.
        Trả về dict: {response, thread_id}
        """
        if not self.agent_graph:
            raise RuntimeError("Agent components are not initialized.")

        tid = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}
        uid = user_id if user_id else "Chưa đăng nhập"

        try:
            inputs = {
                "messages": [
                    ("user", f"[User ID: {uid}] {question}")
                ],
                "next": "",
                "user_id": str(uid)
            }
            response_state = self.agent_graph.invoke(inputs, config)

            final_message = response_state["messages"][-1].content

            products = []
            for msg in reversed(response_state["messages"]):
                if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "check_inventory":
                    try:
                        tool_data = json.loads(msg.content)
                        products = tool_data.get("raw_products", [])
                        break
                    except:
                        pass

            return {
                "response": final_message,
                "thread_id": tid,
                "products": products
            }
        except Exception as e:
            print(f"Multi-Agent execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response": "Dạ hệ thống đang quá tải hoặc gặp lỗi. Anh/chị vui lòng thử lại sau nhé.",
                "thread_id": tid,
            }


# Singleton instance
rag_service = RAGService()
