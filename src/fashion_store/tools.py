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
        JSON string chứa danh sách sản phẩm phù hợp
    """
    try:
        vectorstore = get_vectorstore()
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)
        
        products = []
        for doc, score in docs_and_scores:
            meta = doc.metadata
            products.append({
                "id": meta.get("product_id", "Unknown"),
                "name": meta.get("name", "Unknown"),
                "price": meta.get("price", 0),
                "style": meta.get("style", "Unknown"),
                "similarity_score": round(1 - score, 2)
            })
        
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
    
    # Execute query
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "fashion_password")
    
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
