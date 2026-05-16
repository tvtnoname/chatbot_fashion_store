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
                "pending_agents": [],
                "current_agent_index": 0,
                "agent_responses": [],
                "direct_response": "",
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
        Hỗ trợ Sequential Multi-Routing: stream output từ nhiều sub-agents tuần tự.
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
            "pending_agents": [],
            "current_agent_index": 0,
            "agent_responses": [],
            "direct_response": "",
            "user_id": str(uid),
            "products": []
        }

        products = []
        chunk_count = 0
        # Track tool state PER sub-agent
        tool_was_called = False
        tool_finished = False
        # Track agent hiện tại đang stream
        current_streaming_agent = None   # Tên agent đang active (ops_agent, support_agent, ...)
        current_streaming_node = None    # Node name trong on_chat_model_stream
        # Track xem đã gửi bao nhiêu agent
        agent_stream_count = 0
        # Nodes nội bộ cần bỏ qua khi stream
        SKIP_NODES = {"supervisor", "router_node", "synthesizer_node", "direct_response_node"}
        # Agent nodes
        AGENT_NODES = {"ops_agent", "sales_agent", "support_agent"}
        AGENT_STATUS = {
            "ops_agent": "Đang tra cứu đơn hàng...",
            "sales_agent": "Đang tìm kiếm sản phẩm...",
            "support_agent": "Đang tra cứu chính sách...",
        }
        # ── Fallback: lưu response từ on_chain_end của mỗi agent ──
        agent_responses = {}       # {agent_name: response_text}
        agents_streamed = set()    # Tên agents đã stream thành công ít nhất 1 chunk

        try:
            print(f"    🔄 [Stream] Starting astream_events v2 (Multi-Routing)...")
            async for event in self.agent_graph.astream_events(inputs, config, version="v2"):
                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node", "")

                # ── 0. Bắt direct_response (chào hỏi - không có LLM stream) ──
                if kind == "on_chain_end" and node_name == "direct_response_node":
                    try:
                        output = event.get("data", {}).get("output", {})
                        msgs = output.get("messages", [])
                        if msgs:
                            content = msgs[0].content if hasattr(msgs[0], "content") else str(msgs[0])
                            chunk_count += 1
                            print(f"    👋 [Stream] Direct response sent")
                            yield "data: " + json.dumps({"chunk": content}) + "\n\n"
                    except Exception:
                        pass
                    continue

                # ── 0.5. Gửi status indicator khi bắt đầu agent mới ──
                if kind == "on_chain_start" and node_name in AGENT_NODES:
                    if agent_stream_count > 0:
                        yield "data: " + json.dumps({"chunk": "\n\n"}) + "\n\n"
                    current_streaming_agent = node_name
                    # Reset tool state cho agent mới
                    tool_was_called = False
                    tool_finished = False
                    agent_stream_count += 1
                    status_msg = AGENT_STATUS.get(node_name, "Đang xử lý tiếp...")
                    yield "data: " + json.dumps({"status": status_msg}) + "\n\n"
                    print(f"    🔔 [Stream] Agent starting: {node_name}")
                    continue

                # ── 0.6. Bắt response đầy đủ từ on_chain_end của agent (FALLBACK) ──
                if kind == "on_chain_end" and node_name in AGENT_NODES:
                    try:
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            responses = output.get("agent_responses", [])
                            if responses:
                                agent_responses[node_name] = responses[0]
                                print(f"    📥 [Fallback] Captured {node_name} response ({len(responses[0])} chars)")
                    except Exception:
                        pass
                    continue

                # ── 1. Stream text chunks từ LLM ──
                if kind == "on_chat_model_stream":
                    # Bỏ qua output từ các node nội bộ
                    if node_name in SKIP_NODES:
                        continue

                    # Phát hiện chuyển node mới → reset streaming node
                    if node_name != current_streaming_node:
                        current_streaming_node = node_name
                        print(f"    🔀 [Stream] Switched to node: {node_name}")

                    chunk = event["data"]["chunk"]

                    # Bỏ qua tool-calling chunks
                    if getattr(chunk, "tool_call_chunks", None):
                        continue

                    content = getattr(chunk, "content", None)
                    if not content:
                        continue

                    # Nếu agent đã gọi tool nhưng tool chưa xong → bỏ qua
                    if tool_was_called and not tool_finished:
                        continue

                    # Lọc bỏ JSON thô bị lọt ra
                    stripped = content.strip()
                    if stripped.startswith('{"') or stripped.startswith('[{'):
                        continue
                    # Lọc bỏ raw tool call fragments
                    if any(tn in stripped for tn in ["policy_retriever", "check_inventory", "check_order_status"]):
                        continue

                    chunk_count += 1
                    if current_streaming_agent:
                        agents_streamed.add(current_streaming_agent)
                    if chunk_count == 1:
                        print(f"    ✍️ [Stream] First chunk received! Streaming started...")
                    yield "data: " + json.dumps({"chunk": content}) + "\n\n"

                # ── 2. Tool bắt đầu ──
                elif kind == "on_tool_start":
                    tool_was_called = True
                    tool_finished = False
                    print(f"    🔧 [Stream] Tool started: {event.get('name')}")

                # ── 3. Tool kết thúc ──
                elif kind == "on_tool_end":
                    tool_finished = True
                    print(f"    🔧 [Stream] Tool ended: {event.get('name')}")
                    if event.get("name") == "check_inventory":
                        try:
                            output = event["data"].get("output", "")
                            raw = output.content if hasattr(output, "content") else str(output)
                            tool_data = json.loads(raw)
                            products = tool_data.get("raw_products", [])
                            print(f"    🛒 [Stream] Got {len(products)} products from tool")
                        except Exception:
                            pass

            # ── FALLBACK: Gửi bổ sung response của agents bị miss ──
            for agent_name, response_text in agent_responses.items():
                if agent_name not in agents_streamed and response_text:
                    print(f"    🔄 [Fallback] Sending missed {agent_name} response")
                    # Gửi separator nếu đã có content trước đó
                    if chunk_count > 0:
                        yield "data: " + json.dumps({"chunk": "\n\n"}) + "\n\n"
                    yield "data: " + json.dumps({"chunk": response_text}) + "\n\n"
                    chunk_count += 1

            print(f"    ✅ [Stream] Done. Total chunks sent: {chunk_count}, streamed_agents={agents_streamed}, fallback_agents={set(agent_responses.keys()) - agents_streamed}")
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
