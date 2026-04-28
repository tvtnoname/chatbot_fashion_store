from .agents import fashion_agent
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

def run_fashion_consultant(user_query: str, return_json: bool = False) -> Dict[str, Any]:
    """Execute fashion consultation using LangGraph agent.
    
    Returns:
        Dict with 'response' (str) and 'recommended_product_ids' (list of dicts)
    """
    print(f"\n\n🤖 KHỞI ĐỘNG HỆ THỐNG TƯ VẤN THỜI TRANG...")
    print(f"User Query: {user_query}\n")
    
    # Initialize state
    initial_state = {
        "user_query": user_query,
        "intent_analysis": "",
        "product_search_results": "",
        "recommended_product_ids": [],
        "final_response": ""
    }
    
    # Run the agent graph
    result = fashion_agent.invoke(initial_state)
    
    # Extract results
    final_response = result.get("final_response", "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu.")
    recommended_products = result.get("recommended_product_ids", [])
    
    print(f"\n✅ KẾT QUẢ:\n{final_response}\n")
    print(f"📦 Sản phẩm gợi ý: {len(recommended_products)} sản phẩm")
    
    return {
        "response": final_response,
        "recommended_product_ids": recommended_products
    }

if __name__ == "__main__":
    # Test query
    query = "Gợi ý cho tôi một set đồ đi làm văn phòng"
    result = run_fashion_consultant(query)
    print(f"\nResponse: {result['response']}")
    print(f"Products: {result['recommended_product_ids']}")
