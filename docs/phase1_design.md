# Thiết Kế Chi Tiết Giai Đoạn 1: Hệ Thống Multi-Agent Thời Trang Thông Minh

Tài liệu này mô tả chi tiết thiết kế cho Giai đoạn 1 của hệ thống, tập trung vào Ontology, Quy trình Agent và Môi trường DevOps.

## 1. Thiết kế Ontology (Neo4j Schema)

Sơ đồ tri thức (Knowledge Graph) là trái tim của hệ thống, lưu trữ không chỉ dữ liệu sản phẩm mà còn cả các quy tắc phối đồ và bối cảnh sử dụng.

### 1.1. Khởi tạo Schema (Constraints & Indexes)

Chúng ta cần tạo ràng buộc (Constraint) để đảm bảo tính duy nhất và Index để tăng tốc độ truy vấn.

```cypher
// 1. Product: ID là duy nhất
CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE;

// 2. Tên của các thực thể danh mục nên là duy nhất để tránh trùng lặp
CREATE CONSTRAINT style_name_unique IF NOT EXISTS FOR (s:Style) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT material_name_unique IF NOT EXISTS FOR (m:Material) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT occasion_name_unique IF NOT EXISTS FOR (o:Occasion) REQUIRE o.name IS UNIQUE;
CREATE CONSTRAINT color_name_unique IF NOT EXISTS FOR (c:Color) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT customer_id_unique IF NOT EXISTS FOR (u:Customer) REQUIRE u.id IS UNIQUE;

// 3. Fulltext Index cho tìm kiếm tên sản phẩm và mô tả (Hỗ trợ tìm kiếm từ khóa)
CREATE FULLTEXT INDEX product_search IF NOT EXISTS FOR (n:Product) ON EACH [n.name, n.description];
```

### 1.2. Định nghĩa Node và Properties

| Node Label | Thuộc tính (Properties) | Mô tả |
| :--- | :--- | :--- |
| **Product** | `id` (String), `name` (String), `price` (Float), `stock` (Integer), `image_url` (String), `description` (String), `gender` (String: 'Male', 'Female', 'Unisex') | Thực thể chính, đại diện cho mặt hàng thời trang. |
| **Category** | `name` (String) | Loại sản phẩm (ví dụ: Shirt, Trousers, Dress). |
| **Style** | `name` (String), `vibe` (String) | Phong cách (ví dụ: Vintage, Minimalist, Streetwear). |
| **Material** | `name` (String), `breathability` (String), `texture` (String) | Chất liệu (ví dụ: Cotton, Linen, Silk). |
| **Occasion** | `name` (String), `formality_level` (Integer: 1-10) | Dịp sử dụng (ví dụ: Wedding, Beach, Office). |
| **Color** | `name` (String), `hex_code` (String) | Màu sắc (ví dụ: Navy Blue, Beige). |
| **Customer** | `id` (String), `name` (String), `gender` (String) | Người dùng hệ thống. |

### 1.3. Định nghĩa Relationship

Quan hệ định nghĩa ngữ nghĩa và logic tư vấn (Recommendation Logic).

| Quan hệ (Type) | Từ Node | Đến Node | Thuộc tính quan hệ | Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| **BELONGS_TO** | Product | Category | - | Sản phẩm thuộc loại nào. |
| **HAS_STYLE** | Product | Style | `weight` (Float: 0.0-1.0) | Mức độ phù hợp với phong cách. |
| **MADE_OF** | Product | Material | `percentage` (Integer) | Thành phần chất liệu. |
| **HAS_COLOR** | Product | Color | - | Màu sắc của sản phẩm. |
| **SUITABLE_FOR**| Product | Occasion | `score` (Float), `season` (List<String>) | Độ phù hợp với dịp và mùa (Xuân, Hạ...). |
| **MATCHES_WITH**| Product | Product | `score` (Float), `type` (String: 'Complementary', 'Similar'), `reason` (String) | **Quan trọng**: Logic phối đồ (A hợp với B). |
| **PURCHASED** | Customer| Product | `date` (Datetime), `rating` (Integer) | Lịch sử mua hàng. |

---

## 2. Thiết Kế Luồng Agent (Agentic Workflow)

Hệ thống hoạt động theo mô hình Pipeline, nơi đầu ra của Agent trước là đầu vào của Agent sau.

**Sơ đồ tóm tắt:**
`User Query` -> **Intent Agent** -> `Structured Intent` -> **Graph Query Agent** -> `Cypher/Products` -> **Stylist Agent** -> `Final Advice` -> `User`

### 2.1. Chi tiết vai trò và tương tác

#### 1. Intent Agent (Người thấu hiểu)
*   **Nhiệm vụ**: Phân tích ngôn ngữ tự nhiên, trích xuất thực thể (NER) và xác định mục tiêu người dùng.
*   **Input**: "Tôi cần tìm một bộ đồ lịch sự cho nam, vải mát để đi đám cưới bãi biển."
*   **Process**: Sử dụng LLM (GPT/Gemini) với Prompt chuyên biệt để extract JSON.
*   **Output**:
    ```json
    {
      "intent": "consultation",
      "entities": {
        "gender": "Male",
        "occasion": "Beach Wedding",
        "material_feature": "cool/breathable",
        "style_vibe": "Formal/Smart Casual"
      }
    }
    ```

#### 2. Graph Query Agent (Kỹ sư dữ liệu)
*   **Nhiệm vụ**: Chuyển đổi `Structured Intent` thành truy vấn cơ sở dữ liệu (Cypher cho Neo4j hoặc Vector Search cho ChromaDB).
*   **Logic chuyển đổi**:
    *   Mapping `occasion: "Beach Wedding"` -> Tìm Node `Occasion {name: 'Beach Wedding'}` hoặc Semantic Search vector gần nhất.
    *   Mapping `material_feature: "breathable"` -> Tìm Node `Material` có thuộc tính `breathability: 'High'`.
*   **Sinh lệnh Cypher**:
    ```cypher
    MATCH (p:Product)-[:SUITABLE_FOR]->(o:Occasion)
    WHERE o.name CONTAINS 'Beach' AND p.gender = 'Male'
    MATCH (p)-[:MADE_OF]->(m:Material)
    WHERE m.breathability = 'High' OR m.name IN ['Linen', 'Cotton']
    RETURN p, m, o LIMIT 5
    ```
*   **Output**: Danh sách các Node `Product` tìm được (kèm thông tin chi tiết).

#### 3. Stylist Agent (Nhà tạo mẫu)
*   **Nhiệm vụ**: Nhận danh sách sản phẩm thô, áp dụng kiến thức thời trang để lọc và đề xuất phối đồ (Mix & Match).
*   **Logic**:
    *   Nhận 5 áo sơ mi từ Graph Query Agent.
    *   Với mỗi áo, truy vấn lại Graph (hoặc tra cứu nội bộ) quan hệ `MATCHES_WITH` để tìm quần hoặc phụ kiện đi kèm.
    *   *Ví dụ*: Áo sơ mi Linen trắng (Product A) `MATCHES_WITH` Quần Chinos màu Be (Product B).
*   **Output**: Lời tư vấn hoàn chỉnh.
    > "Dựa trên yêu cầu đi đám cưới bãi biển của bạn, tôi gợi ý áo sơ mi Linen trắng (mã A) vì chất liệu thoáng mát. Bạn nên phối cùng quần Chinos màu Be (mã B) để giữ vẻ lịch sự nhưng vẫn thoải mái. Đừng quên một đôi Loafers nâu!"

---

## 3. Thiết Lập Môi Trường (DevOps)

Cấu trúc dự án sử dụng Docker Compose để quản lý các service.

### 3.1. Docker Compose (`docker-compose.yml`)

File cấu hình này sẽ dựng:
1.  **Neo4j**: Database đồ thị (Admin UI tại port 7474).
2.  **ChromaDB**: Vector Database (API tại port 8000).
3.  **App**: Môi trường Python (Jupyter/API) để chạy code Agent.

```yaml
version: '3.8'

services:
  # 1. Graph Database
  neo4j:
    image: neo4j:5.15.0
    container_name: fashion_neo4j
    ports:
      - "7474:7474" # HTTP (Browser UI)
      - "7687:7687" # Bolt (Driver connection)
    environment:
      NEO4J_AUTH: neo4j/fashion_password
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]' # Cài sẵn APOC và GDS
      NEO4J_dbms_security_procedures_unrestricted: apoc.*,gds.*
    volumes:
      - ./data/neo4j/data:/data
      - ./data/neo4j/logs:/logs
    networks:
      - fashion_net
    healthcheck:
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider localhost:7474 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  # 2. Vector Database
  chromadb:
    image: chromadb/chroma:latest
    container_name: fashion_chroma
    ports:
      - "8000:8000"
    volumes:
      - ./data/chroma:/chroma/chroma
    networks:
      - fashion_net

  # 3. Application / Agent Runtime
  agent_app:
    build: 
      context: .
      dockerfile: Dockerfile
    container_name: fashion_agent_core
    volumes:
      - .:/app
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=fashion_password
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - OPENAI_API_KEY=${OPENAI_API_KEY} # Load from .env
    command: python main.py # Hoặc tail -f /dev/null để giữ container chạy
    depends_on:
      neo4j:
        condition: service_healthy
    networks:
      - fashion_net

networks:
  fashion_net:
    driver: bridge
```

### 3.2. Dockerfile & Python Requirements

**Dockerfile** (để build service `agent_app`):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Cài đặt system dependencies nếu cần
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

CMD ["python", "main.py"]
```

**Thư viện Python quan trọng (`requirements.txt`)**:

```text
# Kết nối Database
neo4j>=5.0.0          # Neo4j Driver
chromadb>=0.4.0       # ChromaDB Client

# AI & Agent Frameworks
langchain             # Orchestration chính
langchain-community
langchain-openai      # Hoặc langchain-google-genai tùy model
crewai                # (Tùy chọn) Nếu muốn quản lý Agent theo Role/Goal rõ ràng
pydantic              # Validation dữ liệu (Output Parser)

# Backend API
fastapi               # Expose API cho Frontend
uvicorn               # Server

# Tiện ích
python-dotenv         # Load biến môi trường
numpy
pandas                # Xử lý dữ liệu dạng bảng trước khi đưa vào Graph
```
