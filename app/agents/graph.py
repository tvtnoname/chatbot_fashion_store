"""
LangGraph Workflow - Đồ thị Multi-Agent chính.
Kết nối Supervisor với các Sub-Agents thành một hệ thống điều phối hoàn chỉnh.
"""
import json
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from app.agents.state import AgentState
from app.agents.supervisor import create_supervisor_node
from app.agents.sub_agents import create_support_agent, create_ops_agent, create_sales_agent


def build_multi_agent_graph():
    """
    Xây dựng và compile đồ thị LangGraph Multi-Agent.
    Luồng đi: User -> Supervisor -> Sub-Agent -> Supervisor -> User
    """
    print("🔄 Building Multi-Agent Graph...")

    # ── Khởi tạo các thành phần ──
    supervisor_node = create_supervisor_node()
    support_agent = create_support_agent()
    ops_agent = create_ops_agent()
    sales_agent = create_sales_agent()

    # ── Wrapper functions cho Sub-Agents ──
    # Các hàm này nhận state, gọi sub-agent, rồi trả kết quả về dạng AIMessage.

    def support_node(state: AgentState) -> AgentState:
        """Gọi Support Agent xử lý câu hỏi chính sách."""
        print("  📋 [Support Agent] Processing...")
        result = support_agent.invoke({"messages": state["messages"]})
        last_msg = result["messages"][-1]
        return {"messages": [AIMessage(content=last_msg.content)]}

    def ops_node(state: AgentState) -> AgentState:
        """Gọi Operations Agent xử lý câu hỏi đơn hàng."""
        print("  📦 [Operations Agent] Processing...")
        result = ops_agent.invoke({"messages": state["messages"]})
        last_msg = result["messages"][-1]
        return {"messages": [AIMessage(content=last_msg.content)]}

    def sales_node(state: AgentState) -> AgentState:
        """Gọi Sales Agent xử lý câu hỏi sản phẩm/tồn kho."""
        print("  🛍️ [Sales Agent] Processing...")
        result = sales_agent.invoke({"messages": state["messages"]})
        last_msg = result["messages"][-1]

        # Extract products from ToolMessages before discarding them
        products = []
        for msg in result["messages"]:
            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "check_inventory":
                try:
                    tool_data = json.loads(msg.content)
                    products = tool_data.get("raw_products", [])
                except Exception:
                    pass

        return {"messages": [AIMessage(content=last_msg.content)], "products": products}

    def greeting_node(state: AgentState) -> AgentState:
        """Trả lời chào hỏi trực tiếp, không cần gọi sub-agent nào."""
        print("  👋 [Direct Response] Greeting...")
        # Lấy tin nhắn gốc của khách để AI chào lại tự nhiên
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage
        llm = ChatOllama(model="llama3.1", temperature=0.7)
        greeting_prompt = SystemMessage(content="""Bạn là AI chăm sóc khách hàng của cửa hàng thời trang. Luôn trả lời bằng tiếng Việt.
Khách hàng đang chào hỏi bạn. Hãy chào lại thân thiện, vui vẻ và gợi ý khách hỏi về sản phẩm, đơn hàng, hoặc chính sách.
Trả lời ngắn gọn trong 2-3 câu. KHÔNG gọi bất kỳ tool nào. KHÔNG trả về JSON.""")
        result = llm.invoke([greeting_prompt, *state["messages"]])
        return {"messages": [AIMessage(content=result.content)]}



    # ── Xây dựng Đồ thị ──
    workflow = StateGraph(AgentState)

    # Thêm các Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("ops_agent", ops_node)
    workflow.add_node("sales_agent", sales_node)
    workflow.add_node("support_agent", support_node)
    workflow.add_node("greeting", greeting_node)


    # Điểm bắt đầu: luôn đi vào Supervisor trước
    workflow.set_entry_point("supervisor")

    # Conditional Edges: Supervisor quyết định đi đâu tiếp
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "ops_agent": "ops_agent",
            "sales_agent": "sales_agent",
            "support_agent": "support_agent",
            "FINISH": "greeting",
        }
    )

    # Sau khi Sub-Agent xử lý xong -> Kết thúc (trả kết quả về cho user)
    workflow.add_edge("ops_agent", END)
    workflow.add_edge("sales_agent", END)
    workflow.add_edge("support_agent", END)
    workflow.add_edge("greeting", END)


    # Compile đồ thị với checkpointer
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    print("✅ Multi-Agent Graph compiled successfully!")

    return graph
