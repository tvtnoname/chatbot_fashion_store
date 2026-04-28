import os
import json
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def vector_search(query: str) -> str:
    """Tìm kiếm sản phẩm thời trang dựa trên mô tả ngữ nghĩa.
    
    Args:
        query: Mô tả sản phẩm cần tìm (ví dụ: 'áo sơ mi trắng đi biển')
    
    Returns:
        JSON string chứa danh sách sản phẩm phù hợp đã được kiểm tra tình trạng active và còn hàng.
    """
    try:
        # P0 FIX: Call Backend's Supabase pgvector endpoint instead of local ChromaDB
        API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:5001/api/v1")
        
        # We append semantic-search path
        url = f"{API_BASE_URL}/products/semantic-search"
        
        # Make request to the backend
        res = requests.get(url, params={"q": query, "limit": 5}, timeout=10.0)
        res.raise_for_status()
        
        data = res.json()
        items = data.get("items", [])
        
        # Map fields back to what the AI Agent expects
        products = []
        for item in items:
            products.append({
                "id": item.get("product_id"),
                "name": item.get("name", "Unknown"),
                "price": item.get("base_price", 0),
                "similarity_score": round(item.get("similarity", 0), 2),
                "stock": item.get("stock", 0)
            })
            
        return json.dumps(products, ensure_ascii=False, indent=2)
    except requests.exceptions.RequestException as e:
        print(f"[vector_search] Network error: {str(e)}")
        return json.dumps([], ensure_ascii=False)
    except Exception as e:
        print(f"[vector_search] Error: {str(e)}")
        return json.dumps([], ensure_ascii=False)
