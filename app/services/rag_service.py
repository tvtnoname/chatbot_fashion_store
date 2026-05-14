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
                "user_id": str(uid),
                "products": []
            }
            response_state = self.agent_graph.invoke(inputs, config)

            final_message = response_state["messages"][-1].content
            products = response_state.get("products", [])

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

    async def astream_answer(self, question: str, user_id: int = None, thread_id: str = None):
        """
        Stream câu trả lời từ AI theo từng chunk nhỏ.
        """
        if not self.agent_graph:
            yield json.dumps({"error": "Agent components are not initialized."}) + "\n\n"
            return

        tid = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}
        uid = user_id if user_id else "Chưa đăng nhập"

        inputs = {
            "messages": [
                ("user", f"[User ID: {uid}] {question}")
            ],
            "next": "",
            "user_id": str(uid),
            "products": []
        }

        products = []
        try:
            async for event in self.agent_graph.astream_events(inputs, config, version="v1"):
                kind = event["event"]
                
                # Bắt các chunk text từ LLM sinh ra
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if getattr(chunk, "content", None):
                        yield json.dumps({"chunk": chunk.content}) + "\n\n"
                        
                # Bắt sự kiện chạy tool xong để lấy danh sách sản phẩm
                elif kind == "on_tool_end":
                    if event.get("name") == "check_inventory":
                        try:
                            tool_data = json.loads(event["data"].get("output", "{}"))
                            products = tool_data.get("raw_products", [])
                        except Exception:
                            pass

            # Sau khi sinh text xong, đẩy chunk cuối chứa metadata (thread_id, products)
            yield json.dumps({
                "thread_id": tid,
                "products": products,
                "is_done": True
            }) + "\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"chunk": "\n[Hệ thống quá tải, vui lòng thử lại sau]"}) + "\n\n"
            yield json.dumps({"thread_id": tid, "products": [], "is_done": True}) + "\n\n"



# Singleton instance
rag_service = RAGService()
