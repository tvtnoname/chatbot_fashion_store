from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from .tools import vector_search, graph_query

# LLM Configuration
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# State Definition
class AgentState(TypedDict):
    user_query: str
    intent_analysis: str
    product_search_results: str
    graph_query_results: str
    final_response: str

# Node 1: Intent Analysis
def intent_analysis_node(state: AgentState) -> AgentState:
    """Phân tích ý định người dùng"""
    query = state["user_query"]
    
    messages = [
        SystemMessage(content=(
            "Bạn là chuyên gia phân tích nhu cầu thời trang. "
            "Phân tích yêu cầu người dùng và trích xuất: "
            "core_intent (món đồ cần), occasion (dịp), style_vibe (phong cách), "
            "color_preference (màu), material_preference (chất liệu). "
            "Trả về JSON."
        )),
        HumanMessage(content=f"Yêu cầu: {query}")
    ]
    
    response = llm.invoke(messages)
    state["intent_analysis"] = response.content
    return state

# Node 2: Product Search  
def product_search_node(state: AgentState) -> AgentState:
    """Tìm kiếm sản phẩm phù hợp"""
    intent = state["intent_analysis"]
    query = state["user_query"]
    
    # Invoke vector search tool
    search_result = vector_search.invoke({"query": query})
    state["product_search_results"] = search_result
    
    return state

# Node 3: Graph Query (Optional - for matching items)
def graph_knowledge_node(state: AgentState) -> AgentState:
    """Tìm sản phẩm phối hợp từ Knowledge Graph"""
    products = state.get("product_search_results", "")
    
    if products and products != "[]":
        # Extract first product name to find matches
        try:
            import json
            product_list = json.loads(products)
            if product_list:
                first_product = product_list[0]["name"]
                graph_result = graph_query.invoke({
                    "query": f"Tìm sản phẩm phối hợp với {first_product}"
                })
                state["graph_query_results"] = graph_result
        except:
            state["graph_query_results"] = "[]"
    else:
        state["graph_query_results"] = "[]"
    
    return state

# Node 4: Final Response Generation
def stylist_response_node(state: AgentState) -> AgentState:
    """Tạo lời khuyên stylist bằng tiếng Việt"""
    messages = [
        SystemMessage(content=(
            "Bạn là chuyên gia tư vấn thời trang cao cấp tại Việt Nam. "
            "QUAN TRỌNG: "
            "1. Trả lời HOÀN TOÀN BẰNG TIẾNG VIỆT, văn phong tự nhiên như đang trò chuyện. "
            "2. KHÔNG dùng markdown (**, *, #, -). Viết câu văn liền mạch như ChatGPT. "
            "3. SỬ DỤNG EMOJI phù hợp để làm câu trả lời sinh động hơn (👔 áo, 👗 váy, 👞 giày, 👜 túi, 💼 công sở, ✨ nổi bật, 🎨 màu sắc, etc.). "
            "4. Phong cách: Thân thiện, chuyên nghiệp, như một người bạn am hiểu thời trang. "
            "5. Không liệt kê dạng bullet points. Hãy kể câu chuyện về outfit. "
            "6. Giải thích TẠI SAO các món đồ hợp với nhau, dựa trên màu sắc, chất liệu, dịp. "
            "7. Nếu có sản phẩm cụ thể từ database, hãy tự nhiên đề cập tên và giá."
        )),
        HumanMessage(content=f"""Yêu cầu: {state['user_query']}

Phân tích nhu cầu: {state['intent_analysis']}

Sản phẩm tìm được: {state['product_search_results']}

Sản phẩm phối hợp: {state.get('graph_query_results', '[]')}

Hãy tư vấn một outfit hoàn chỉnh, viết tự nhiên như đang nhắn tin với bạn bè.""")
    ]
    
    response = llm.invoke(messages)
    state["final_response"] = response.content
    return state

# Build Graph
def create_fashion_agent():
    """Tạo LangGraph agent cho tư vấn thời trang"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("intent_analysis", intent_analysis_node)
    workflow.add_node("product_search", product_search_node)
    workflow.add_node("graph_knowledge", graph_knowledge_node)
    workflow.add_node("stylist_response", stylist_response_node)
    
    # Define edges
    workflow.set_entry_point("intent_analysis")
    workflow.add_edge("intent_analysis", "product_search")
    workflow.add_edge("product_search", "graph_knowledge")
    workflow.add_edge("graph_knowledge", "stylist_response")
    workflow.add_edge("stylist_response", END)
    
    return workflow.compile()

# Export compiled agent
fashion_agent = create_fashion_agent()
