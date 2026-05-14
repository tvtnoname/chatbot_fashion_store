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
        Chỉ stream output từ sub-agent (bỏ supervisor, bỏ tool-calling chunks).
        """
        if not self.agent_graph:
            yield "data: " + json.dumps({"error": "Agent components are not initialized."}) + "\n\n"
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
        chunk_count = 0
        # Track xem tool đã được gọi chưa (trong mỗi sub-agent)
        tool_was_called = False
        # Track xem tool đã chạy xong chưa → chỉ stream LLM output SAU khi tool xong
        tool_finished = False

        try:
            print(f"    🔄 [Stream] Starting astream_events v2...")
            async for event in self.agent_graph.astream_events(inputs, config, version="v2"):
                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node", "")

                # ── 1. Stream text chunks từ LLM ──
                if kind == "on_chat_model_stream":
                    # BỎ QUA output từ supervisor (nó chỉ output tên agent như "sales_agent")
                    if node_name == "supervisor":
                        continue

                    chunk = event["data"]["chunk"]

                    # Bỏ qua tool-calling chunks (LLM quyết định gọi tool, không phải câu trả lời)
                    if getattr(chunk, "tool_call_chunks", None):
                        continue

                    content = getattr(chunk, "content", None)
                    if not content:
                        continue

                    # Nếu agent đã gọi tool nhưng tool chưa xong → đây là LLM đang ra lệnh gọi tool, bỏ qua
                    if tool_was_called and not tool_finished:
                        continue

                    # Lọc bỏ JSON thô bị lọt ra (ví dụ: {"name": "..."})
                    stripped = content.strip()
                    if stripped.startswith('{"') or stripped.startswith('[{'):
                        continue

                    chunk_count += 1
                    if chunk_count == 1:
                        print(f"    ✍️ [Stream] First chunk received! Streaming started...")
                    yield "data: " + json.dumps({"chunk": content}) + "\n\n"

                # ── 2. Thông báo trạng thái khi tool bắt đầu (gửi riêng, không nối vào text) ──
                elif kind == "on_tool_start":
                    tool_was_called = True
                    tool_finished = False
                    tool_name = event.get("name")
                    status_msg = None
                    if tool_name == "check_inventory":
                        status_msg = "Đang kiểm tra kho hàng..."
                    elif tool_name == "check_order_status":
                        status_msg = "Đang tra cứu hệ thống..."
                    elif tool_name == "policy_retriever":
                        status_msg = "Đang tra cứu chính sách..."
                    if status_msg:
                        print(f"    🔧 [Stream] Tool started: {tool_name}")
                        yield "data: " + json.dumps({"status": status_msg}) + "\n\n"

                # ── 3. Tool kết thúc → cho phép stream LLM output tiếp theo ──
                elif kind == "on_tool_end":
                    tool_finished = True
                    if event.get("name") == "check_inventory":
                        try:
                            output = event["data"].get("output", "")
                            raw = output.content if hasattr(output, "content") else str(output)
                            tool_data = json.loads(raw)
                            products = tool_data.get("raw_products", [])
                            print(f"    🛒 [Stream] Got {len(products)} products from tool")
                        except Exception:
                            pass

            print(f"    ✅ [Stream] Done. Total chunks sent: {chunk_count}")
            yield "data: " + json.dumps({
                "thread_id": tid,
                "products": products,
                "is_done": True
            }) + "\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
            yield "data: " + json.dumps({"chunk": "\n[Hệ thống quá tải, vui lòng thử lại sau]"}) + "\n\n"
            yield "data: " + json.dumps({"thread_id": tid, "products": [], "is_done": True}) + "\n\n"



# Singleton instance
rag_service = RAGService()
