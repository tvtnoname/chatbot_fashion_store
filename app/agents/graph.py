"""
LangGraph Workflow - Đồ thị Multi-Agent chính.
Kết nối Supervisor với các Sub-Agents thành hệ thống điều phối tuần tự (Sequential Multi-Routing).
Hỗ trợ Intent Splitting: một câu hỏi có thể được giao cho nhiều agent xử lý lần lượt.
"""
import json
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.agents.supervisor import create_supervisor_node
from app.agents.sub_agents import create_support_agent, create_ops_agent, create_sales_agent


def build_multi_agent_graph():
    """
    Xây dựng và compile đồ thị LangGraph Multi-Agent với Sequential Multi-Routing.

    Luồng đi:
      - Chào hỏi: User -> Supervisor (direct_response) -> direct_response_node -> END
      - Câu đơn:  User -> Supervisor -> router -> sub_agent -> router -> synthesizer -> END
      - Câu kép:  User -> Supervisor -> router -> agent1 -> router -> agent2 -> router -> synthesizer -> END
    """
    print("🔄 Building Multi-Agent Graph (Sequential Multi-Routing)...")

    # ── Khởi tạo các thành phần ──
    supervisor_node = create_supervisor_node()
    support_agent = create_support_agent()
    ops_agent = create_ops_agent()
    sales_agent = create_sales_agent()

    # ── Helper: Lấy N cặp hỏi/đáp gần nhất + tin nhắn hiện tại ──
    def _get_recent_messages(messages, max_pairs=2):
        """Chỉ lấy vài cặp hỏi/đáp gần nhất để giữ ngữ cảnh ngắn hạn,
        tránh nhiễu ngữ cảnh xuyên agent (cross-contamination)."""
        if not messages:
            return messages
        filtered = [m for m in messages if getattr(m, "type", "") in ("human", "ai")]
        limit = max_pairs * 2 + 1
        return filtered[-limit:] if len(filtered) > limit else filtered

    def _sanitize_output(content: str) -> str:
        """Nếu AI output JSON thô hoặc raw tool call thay vì câu văn, lọc bỏ phần bị lỗi."""
        stripped = content.strip()
        # Pattern 1: JSON tool call chuẩn → bỏ qua hoàn toàn
        if '{"name":' in stripped and '"parameters":' in stripped:
            return ""
        # Pattern 2: JSON tool call bị malformed (thiếu dấu {, bị cắt)
        tool_names = ["policy_retriever", "check_inventory", "check_order_status"]
        for tool_name in tool_names:
            if f'"{tool_name}"' in stripped and '"parameters"' in stripped:
                # Cố gắng giữ lại phần text trước đoạn JSON bị lỗi
                idx = stripped.find(f'"{tool_name}"')
                search_start = max(0, idx - 50)
                for marker in ['{"name"', '{"', 'name":', '{']:
                    pos = stripped.find(marker, search_start)
                    if pos != -1 and pos < idx:
                        clean = stripped[:pos].strip()
                        if clean:
                            return clean
                        break
                return ""
        return content

    # ══════════════════════════════════════════════════
    # NODE DEFINITIONS
    # ══════════════════════════════════════════════════

    def direct_response_node(state: AgentState) -> AgentState:
        """Đẩy direct_response từ Supervisor vào messages (không gọi LLM)."""
        print("  👋 [Direct Response] Returning supervisor's greeting")
        return {"messages": [AIMessage(content=state["direct_response"])]}

    def router_node(state: AgentState) -> AgentState:
        """Node trung gian - không thay đổi state, chỉ để conditional edge quyết định."""
        idx = state.get("current_agent_index", 0)
        pending = state.get("pending_agents", [])
        remaining = len(pending) - idx
        print(f"  🔀 [Router] index={idx}, pending={pending}, remaining={remaining}")
        return {}

    async def support_node(state: AgentState) -> AgentState:
        """Gọi Support Agent xử lý câu hỏi chính sách."""
        print("  📋 [Support Agent] Processing...")
        recent = _get_recent_messages(state["messages"])
        result = await support_agent.ainvoke({"messages": recent})
        last_msg = result["messages"][-1]
        content = _sanitize_output(last_msg.content)
        idx = state.get("current_agent_index", 0)
        return {
            "agent_responses": [content],
            "current_agent_index": idx + 1,
        }

    async def ops_node(state: AgentState) -> AgentState:
        """Gọi Operations Agent xử lý câu hỏi đơn hàng."""
        print("  📦 [Operations Agent] Processing...")
        recent = _get_recent_messages(state["messages"])
        result = await ops_agent.ainvoke({"messages": recent})
        last_msg = result["messages"][-1]
        content = _sanitize_output(last_msg.content)
        idx = state.get("current_agent_index", 0)
        return {
            "agent_responses": [content],
            "current_agent_index": idx + 1,
        }

    async def sales_node(state: AgentState) -> AgentState:
        """Gọi Sales Agent xử lý câu hỏi sản phẩm/tồn kho."""
        print("  🛍️ [Sales Agent] Processing...")
        recent = _get_recent_messages(state["messages"])
        result = await sales_agent.ainvoke({"messages": recent})
        last_msg = result["messages"][-1]
        content = _sanitize_output(last_msg.content)

        # Extract products from ToolMessages before discarding them
        products = []
        for msg in result["messages"]:
            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "check_inventory":
                try:
                    tool_data = json.loads(msg.content)
                    products = tool_data.get("raw_products", [])
                except Exception:
                    pass

        idx = state.get("current_agent_index", 0)
        return {
            "agent_responses": [content],
            "current_agent_index": idx + 1,
            "products": products,
        }

    def synthesizer_node(state: AgentState) -> AgentState:
        """Gộp kết quả từ nhiều agent thành một câu trả lời thống nhất (template, không gọi LLM)."""
        responses = state.get("agent_responses", [])
        pending = state.get("pending_agents", [])

        print(f"  🔗 [Synthesizer] Merging {len(responses)} response(s)...")

        if len(responses) == 0:
            return {"messages": [AIMessage(
                content="Dạ, hệ thống chưa nhận được kết quả. Anh/chị vui lòng thử lại nhé! 🙏"
            )]}

        # Lọc bỏ responses rỗng (do _sanitize_output trả về "")
        valid = [(i, r) for i, r in enumerate(responses) if r and r.strip()]

        if len(valid) == 0:
            return {"messages": [AIMessage(
                content="Dạ, hệ thống chưa nhận được kết quả. Anh/chị vui lòng thử lại nhé! 🙏"
            )]}

        if len(valid) == 1:
            # Câu đơn - không cần merge
            return {"messages": [AIMessage(content=valid[0][1])]}

        # Câu kép - merge bằng template
        agent_labels = {
            "ops_agent": "📦 **Về đơn hàng của bạn:**",
            "sales_agent": "🛍️ **Về sản phẩm:**",
            "support_agent": "📋 **Về chính sách:**",
        }

        parts = []
        for i, resp in valid:
            label = agent_labels.get(pending[i], "") if i < len(pending) else ""
            parts.append(f"{label}\n{resp}" if label else resp)

        merged = "\n\n".join(parts)
        return {"messages": [AIMessage(content=merged)]}

    # ══════════════════════════════════════════════════
    # ROUTING FUNCTIONS (Conditional Edges)
    # ══════════════════════════════════════════════════

    def route_after_supervisor(state: AgentState) -> str:
        """Sau Supervisor: agents được ưu tiên trước direct_response."""
        if state.get("pending_agents"):
            return "router"
        if state.get("direct_response"):
            return "direct_response"
        return "direct_response"  # fallback

    def route_next_agent(state: AgentState) -> str:
        """Sau Router: chọn agent tiếp theo trong hàng đợi hoặc đi tới synthesizer."""
        idx = state.get("current_agent_index", 0)
        pending = state.get("pending_agents", [])
        if idx < len(pending):
            return pending[idx]
        return "synthesizer"

    # ══════════════════════════════════════════════════
    # XÂY DỰNG ĐỒ THỊ
    # ══════════════════════════════════════════════════
    workflow = StateGraph(AgentState)

    # Thêm các Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("direct_response_node", direct_response_node)
    workflow.add_node("router_node", router_node)
    workflow.add_node("ops_agent", ops_node)
    workflow.add_node("sales_agent", sales_node)
    workflow.add_node("support_agent", support_node)
    workflow.add_node("synthesizer_node", synthesizer_node)

    # Điểm bắt đầu: luôn đi vào Supervisor trước
    workflow.set_entry_point("supervisor")

    # Sau Supervisor: đi tới direct_response hoặc router
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "direct_response": "direct_response_node",
            "router": "router_node",
        }
    )

    # Sau Router: đi tới agent tiếp theo hoặc synthesizer
    workflow.add_conditional_edges(
        "router_node",
        route_next_agent,
        {
            "ops_agent": "ops_agent",
            "sales_agent": "sales_agent",
            "support_agent": "support_agent",
            "synthesizer": "synthesizer_node",
        }
    )

    # Sau mỗi Sub-Agent: quay lại Router (vòng lặp tuần tự)
    workflow.add_edge("ops_agent", "router_node")
    workflow.add_edge("sales_agent", "router_node")
    workflow.add_edge("support_agent", "router_node")

    # Điểm kết thúc
    workflow.add_edge("direct_response_node", END)
    workflow.add_edge("synthesizer_node", END)

    # Compile đồ thị với checkpointer
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    print("✅ Multi-Agent Graph compiled successfully! (Sequential Multi-Routing)")

    return graph
