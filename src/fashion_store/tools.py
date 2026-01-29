import os
import json
from typing import Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
import chromadb
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- ChromaDB Setup (Singleton) ---
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore:
        return _vectorstore
    
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", 8000))
    http_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    from langchain_community.vectorstores import Chroma
    _vectorstore = Chroma(
        client=http_client,
        collection_name="product_descriptions_gemini",
        embedding_function=embedding_model
    )
    return _vectorstore

# --- 1. Vector Search Tool ---
@tool
def vector_search(query: str) -> str:
    """Tìm kiếm sản phẩm thời trang dựa trên mô tả ngữ nghĩa.
    
    Args:
        query: Mô tả sản phẩm cần tìm (ví dụ: 'áo sơ mi trắng đi biển')
    
    Returns:
        JSON string chứa danh sách sản phẩm phù hợp đã được kiểm tra tình trạng active
    """
    try:
        vectorstore = get_vectorstore()
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=10) # Fetch more to filter down
        
        products = []
        import requests
        
        # Helper to check product status via API
        # Assuming backend is accessible at localhost:8000
        # In Docker, this might need 'http://backend:8000'
        # defaulting to localhost check for local dev environment
        API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")
        
        for doc, score in docs_and_scores:
            meta = doc.metadata
            p_id = meta.get("product_id")
            
            if not p_id:
                continue
                
            # --- STATUS CHECK ---
            try:
                # We simply call the public API which now filters by active/stock by default
                # Or we can check specific details if we had an internal verify endpoint.
                # For efficiency, we might want to batch check, but for now per-item check 
                # ensures we get live stock status.
                # Note: Calling /products/{id} might return 404 if active logic was applied there too,
                # OR we might need to inspect the payload.
                # Let's check `read_products` (list) vs `read_product` (detail).
                # `read_product` (detail) uses Supabase directly and doesn't seem to enforce active check in previous code view.
                # So we should ideally filter by calling the LIST endpoint with ID or checking detail manually.
                
                # Fast check using requests
                # Let's just fetch the detail and check manually to be safe
                res = requests.get(f"{API_BASE_URL}/products/{p_id}", timeout=2)
                
                if res.status_code == 200:
                    data = res.json()
                    product_info = data.get("product", {})
                    total_stock = data.get("total_stock", 0)
                    is_active = product_info.get("is_active", False)
                    
                    if is_active and total_stock > 0:
                        products.append({
                            "id": p_id,
                            "name": meta.get("name", "Unknown"),
                            "price": meta.get("price", 0),
                            "style": meta.get("style", "Unknown"),
                            "similarity_score": round(1 - score, 2),
                            "stock": total_stock # Useful context for AI
                        })
            except Exception as e:
                # If API fail, we might skip to be safe, or include with warning. 
                # Let's skip to ensure high quality results.
                # print(f"Check failed for {p_id}: {e}")
                pass
            
            if len(products) >= 5: # Stop once we have 5 valid items
                break
        
        return json.dumps(products, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# --- 2. Graph Query Tool ---
SCHEMA_CONTEXT = """
Node Labels:
- Product (id, name, price, gender)
- Style (name)
- Material (name)
- Occasion (name)
- Color (name)

Relationship Types:
- (:Product)-[:BELONGS_TO]->(:Category)
- (:Product)-[:HAS_STYLE]->(:Style)
- (:Product)-[:MADE_OF]->(:Material)
- (:Product)-[:SUITABLE_FOR]->(:Occasion)
- (:Product)-[:HAS_COLOR]->(:Color)
- (:Product)-[:MATCHES_WITH {reason: '...'}]->(:Product)

Rules:
1. Để tìm sản phẩm phối hợp: MATCH (p:Product {name: '...'})-[:MATCHES_WITH]-(related:Product) RETURN related
2. Để tìm theo dịp: MATCH (p:Product)-[:SUITABLE_FOR]->(o:Occasion {name: '...'}) RETURN p
3. Luôn giới hạn kết quả (LIMIT 5)
4. Trả về tên sản phẩm, giá và lý do (nếu có)
5. Sử dụng CONTAINS để tìm kiếm không phân biệt hoa/thường
"""

@tool
def graph_query(query: str) -> str:
    """Truy vấn Knowledge Graph để tìm quy tắc phối đồ và lọc sản phẩm theo dịp/phong cách.
    
    Args:
        query: Câu hỏi về mối quan hệ thời trang (ví dụ: 'Tìm phụ kiện hợp với áo sơ mi trắng')
    
    Returns:
        JSON string chứa kết quả truy vấn
    """
    # Generate Cypher query
    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là Neo4j Expert. Chuyển câu hỏi thành Cypher Query.\\nSchema:\\n{schema}"),
        ("user", "Câu hỏi: {query}\\n\\nChỉ trả về câu lệnh Cypher, không giải thích.")
    ])
    
    chain = cypher_prompt | llm | StrOutputParser()
    cypher_query = chain.invoke({"schema": SCHEMA_CONTEXT, "query": query}).strip()
    cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    
    # P0 Security: Require environment variables, no hardcoded defaults
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([uri, user, password]):
        return json.dumps({
            "error": "Neo4j configuration missing. Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables."
        }, ensure_ascii=False)
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [r.data() for r in result]
        driver.close()
        
        return json.dumps({
            "cypher": cypher_query,
            "results": records
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: {str(e)}\\nQuery: {cypher_query}"
