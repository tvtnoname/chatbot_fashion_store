import os
import json
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import AI/LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import Databases
from neo4j import GraphDatabase
import chromadb
# from chromadb.utils import embedding_functions # Not strictly needed if using LangChain embedding function adapter

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- 1. Data Models (Pydantic) ---
class ProductAttributes(BaseModel):
    style: str = Field(description="Phong cách thời trang, ví dụ: Vintage, Minimalist, Streetwear, Casual, Formal")
    material: str = Field(description="Chất liệu chính, ví dụ: Cotton, Linen, Silk, Leather, Denim")
    occasion: str = Field(description="Dịp sử dụng phù hợp nhất, ví dụ: Beach Wedding, Office, Casual Street, Gala Dinner")
    color: str = Field(description="Màu sắc chủ đạo, ví dụ: White, Navy Blue, Black, Beige")

# --- 2. LLM Auto-Tagging ---
class LLMTagger:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.warning("GOOGLE_API_KEY chưa được set. Auto-tagging có thể thất bại.")
        
        # Switched to Gemini 2.0 Flash
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=ProductAttributes)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một chuyên gia thời trang. Nhiệm vụ của bạn là trích xuất các thuộc tính thời trang từ mô tả sản phẩm."),
            ("user", "Mô tả sản phẩm: {description}\n\n{format_instructions}")
        ])

    def extract_attributes(self, description: str) -> dict:
        """Sử dụng LLM để phân tích mô tả và trả về JSON."""
        try:
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({
                "description": description,
                "format_instructions": self.parser.get_format_instructions()
            })
            logging.info(f"Đã trích xuất attributes: {result}")
            return result
        except Exception as e:
            logging.error(f"Lỗi khi tagging: {e}")
            # Trả về default nếu lỗi
            return {"style": "Casual", "material": "Unknown", "occasion": "Everyday", "color": "Multi"}

# --- 3. Graph Ingestion (Neo4j) ---
class GraphIngestor:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "fashion_password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ingest_product(self, product: dict, attributes: dict):
        """Đẩy thông tin sản phẩm và thuộc tính vào Neo4j."""
        query = """
        MERGE (p:Product {id: $id})
        SET p.name = $name, 
            p.price = toFloat($price), 
            p.description = $desc, 
            p.image_url = $image_url

        // Tạo Node Thuộc tính nếu chưa có
        MERGE (s:Style {name: $style})
        MERGE (m:Material {name: $material})
        MERGE (o:Occasion {name: $occasion})
        MERGE (c:Color {name: $color})

        // Tạo quan hệ
        MERGE (p)-[:HAS_STYLE]->(s)
        MERGE (p)-[:MADE_OF]->(m)
        MERGE (p)-[:SUITABLE_FOR]->(o)
        MERGE (p)-[:HAS_COLOR]->(c)
        """
        
        params = {
            "id": product["id"],
            "name": product["name"],
            "price": product["price"],
            "desc": product["description"],
            "image_url": product.get("image_url", ""),
            "style": attributes.get("style", "Unknown"),
            "material": attributes.get("material", "Unknown"),
            "occasion": attributes.get("occasion", "Unknown"),
            "color": attributes.get("color", "Unknown")
        }
        
        try:
            with self.driver.session() as session:
                session.run(query, params)
                logging.info(f"Đã nạp Product {product['id']} vào Neo4j.")
        except Exception as e:
            logging.error(f"Lỗi Neo4j Ingestion: {e}")

    def create_rule_based_relationships(self):
        """Tạo quan hệ MATCHES_WITH dựa trên luật đơn giản."""
        query = """
        MATCH (p1:Product)-[:HAS_STYLE]->(s:Style)<-[:HAS_STYLE]-(p2:Product)
        WHERE p1.id < p2.id  // Tránh duplicates
        AND NOT (p1)-[:MATCHES_WITH]-(p2) // Chỉ tạo nếu chưa có
        MERGE (p1)-[:MATCHES_WITH {type: 'Similar Style', score: 0.8}]->(p2)
        RETURN count(*) as created_count
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                count = result.single()["created_count"]
                logging.info(f"Đã tạo {count} quan hệ MATCHES_WITH dựa trên Style.")
        except Exception as e:
             logging.error(f"Lỗi Neo4j Relationship Creation: {e}")

# --- 4. Vector Store (ChromaDB) ---
class VectorStore:
    def __init__(self):
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", 8000))
        self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        
        # Switched to Gemini Embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
             logging.warning("GOOGLE_API_KEY chưa được set. Vector Embedding có thể thất bại.")
        
        # Adapter class to make LangChain embeddings compatible with ChromaDB
        class ChromaEmbeddingAdapter:
            def __init__(self, langchain_embeddings):
                self.lc_embeddings = langchain_embeddings
            def __call__(self, input: List[str]) -> List[List[float]]:
                return self.lc_embeddings.embed_documents(input)
            def name(self):
                return "google_gemini"

        self.lc_embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.embedding_fn = ChromaEmbeddingAdapter(self.lc_embed_model)
        
        # Note: We pass the embedding function directly to get_or_create_collection
        # creating a collection requires an embedding function that expects text input
        self.collection = self.client.get_or_create_collection(
            name="product_descriptions_gemini", # New collection for Gemini
            embedding_function=self.embedding_fn
        )

    def add_product(self, product_id: str, text: str, metadata: dict = None):
        """Lưu vector của mô tả sản phẩm."""
        if metadata is None:
            metadata = {}
        metadata["product_id"] = product_id
        
        try:
            self.collection.add(
                ids=[product_id],
                documents=[text],
                metadatas=[metadata]
            )
            logging.info(f"Đã lưu vector cho Product {product_id} vào ChromaDB.")
        except Exception as e:
            logging.error(f"Lỗi Vector Store: {e}")

    def search_similar(self, query: str, k: int = 3):
        """Tìm kiếm vector tương đồng."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            return results
        except Exception as e:
             logging.error(f"Lỗi Search Vector: {e}")
             return None

# --- 5. Main ETL Pipeline ---
def run_pipeline():
    # Load data
    with open('data/raw/products.json', 'r') as f:
        products = json.load(f)

    tagger = LLMTagger()
    graph_db = GraphIngestor()
    vector_db = VectorStore()

    try:
        for p in products:
            logging.info(f"--- Đang xử lý: {p['name']} ---")
            
            # 1. AI Tagging
            attrs = tagger.extract_attributes(p["description"])
            
            # 2. Graph Ingestion
            graph_db.ingest_product(p, attrs)
            
            # 3. Vector Ingestion
            vector_metadata = attrs.copy()
            vector_metadata["price"] = p["price"]
            vector_metadata["name"] = p["name"]
            
            vector_db.add_product(
                product_id=p["id"],
                text=p["description"],
                metadata=vector_metadata
            )

        # 4. Post-processing
        graph_db.create_rule_based_relationships()

    finally:
        graph_db.close()
        print("\n=== PIPELINE HOÀN TẤT ===")

# --- 6. Hybrid Search Function (Demo) ---
def hybrid_search_demo(query: str):
    print(f"\n>>> Thực hiện Hybrid Search cho: '{query}'")
    
    vector_db = VectorStore()
    graph_db = GraphIngestor()
    
    results = vector_db.search_similar(query, k=3)
    
    if not results or not results['ids']:
        print("Không tìm thấy kết quả vector nào.")
        return

    found_ids = results['ids'][0]
    distances = results['distances'][0]
    
    print("\n[ChromaDB] Tìm thấy các sản phẩm tương đồng:")
    for pid, dist in zip(found_ids, distances):
        print(f"- ID: {pid} (Distance: {dist:.4f})")

    print("\n[Neo4j] Lấy thông tin chi tiết và gợi ý phối đồ:")
    cypher_query = """
    MATCH (p:Product)
    WHERE p.id IN $ids
    OPTIONAL MATCH (p)-[:MATCHES_WITH]-(related:Product)
    RETURN p.name, p.price, collect(related.name) as recommendations
    """
    
    try:
        with graph_db.driver.session() as session:
            records = session.run(cypher_query, ids=found_ids)
            for record in records:
                print(f"* Sản phẩm: {record['p.name']} - ${record['p.price']}")
                if record['recommendations']:
                    print(f"  -> Gợi ý phối cùng: {', '.join(record['recommendations'])}")
                else:
                    print("  -> Chưa có gợi ý phối đồ.")
    except Exception as e:
        print(f"Lỗi truy vấn Graph: {e}")
    finally:
        graph_db.close()

if __name__ == "__main__":
    run_pipeline()
    # hybrid_search_demo("something for a summer beach party")
