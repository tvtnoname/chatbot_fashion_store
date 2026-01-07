from .agents import fashion_agent
from dotenv import load_dotenv

load_dotenv()

def run_fashion_consultant(user_query: str, return_json: bool = False):
    """Execute fashion consultation using LangGraph agent"""
    print(f"\n\n🤖 KHỞI ĐỘNG HỆ THỐNG TƯ VẤN THỜI TRANG...")
    print(f"User Query: {user_query}\n")
    
    # Initialize state
    initial_state = {
        "user_query": user_query,
        "intent_analysis": "",
        "product_search_results": "",
        "graph_query_results": "",
        "final_response": ""
    }
    
    # Run the agent graph
    result = fashion_agent.invoke(initial_state)
    
    # Extract final response
    final_response = result.get("final_response", "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu.")
    
    print(f"\n✅ KẾT QUẢ:\n{final_response}\n")
    
    return final_response

if __name__ == "__main__":
    # Test query
    query = "Gợi ý cho tôi một set đồ đi làm văn phòng"
    print(run_fashion_consultant(query))
