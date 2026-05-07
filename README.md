# Fashion Store RAG Chatbot API

Đây là hệ thống Chatbot tư vấn chính sách cho cửa hàng thời trang. Dự án được cấu trúc theo chuẩn FastAPI và sử dụng mô hình RAG (Retrieval-Augmented Generation) để trả lời câu hỏi dựa trên bộ dữ liệu quy định của cửa hàng, đảm bảo tính chính xác cao và không bịa đặt (Zero Hallucination).

## 🚀 Công nghệ sử dụng
- **Backend Framework**: FastAPI
- **LLM**: Llama3 (chạy local qua Ollama)
- **Vector Database**: ChromaDB
- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (tối ưu cho tiếng Việt)
- **Tunnelling**: Ngrok (tạo public URL cho Frontend)

## 📁 Cấu trúc thư mục

```text
chatbot_fashion_store/
├── .env                  # Biến môi trường (không đẩy lên git)
├── .gitignore            # Các file bị loại trừ khỏi git
├── requirements.txt      # Danh sách thư viện Python
│
├── data/                 # Tầng Dữ Liệu
│   ├── raw/              # Dữ liệu thô (quy_dinh_cua_hang_thoi_trang.json)
│   └── chroma_db/        # Vector Database (sẽ tự động sinh ra)
│
├── scripts/              # Các công cụ hỗ trợ
│   ├── ingest.py         # Script băm nhỏ dữ liệu và nạp vào ChromaDB
│   └── chat_cli.py       # Script test chatbot trực tiếp trên Terminal
│
└── app/                  # Tầng Ứng Dụng Chính
    ├── __init__.py
    ├── main.py           # Khởi tạo FastAPI App
    ├── api/              # Định nghĩa API Endpoints
    ├── core/             # Quản lý cấu hình
    ├── schemas/          # Định nghĩa DTOs (Pydantic models)
    └── services/         # Chứa Logic xử lý (RAG Service)
```

## ⚙️ Cài đặt & Môi trường

### 1. Yêu cầu hệ thống
- **Python**: Phiên bản 3.10+
- **Ollama**: Đã cài đặt và tải model Llama3 (`ollama run llama3`)
- **Ngrok**: Đã cài đặt (`brew install ngrok`)

### 2. Thiết lập môi trường ảo
Khuyến nghị sử dụng môi trường ảo `.venv` để tránh xung đột thư viện:

```bash
# Clone dự án
git clone https://github.com/your-username/chatbot_fashion_store.git
cd chatbot_fashion_store

# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo (Mac/Linux)
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

## 🛠 Hướng dẫn sử dụng hàng ngày

### Bước 1: Khởi động Vector Database (Nạp dữ liệu)
Nếu bạn thay đổi file `data/raw/quy_dinh_cua_hang_thoi_trang.json`, bạn cần chạy lại lệnh sau để nạp dữ liệu vào ChromaDB:
```bash
python scripts/ingest.py
```

### Bước 2: Khởi động API Server
Mở Terminal 1 và chạy lệnh:
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Server sẽ chạy tại `http://localhost:8000`. Bạn có thể truy cập `http://localhost:8000/docs` để xem tài liệu Swagger UI tự động của FastAPI.

### Bước 3: Public API bằng Ngrok (cho Frontend Vercel)
Mở Terminal 2 và chạy lệnh:
```bash
ngrok http 8000
```
Copy dòng `Forwarding` (ví dụ: `https://xxxx.ngrok-free.app`) và điền vào biến môi trường API tương ứng trên Vercel.

## 🧪 Test API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Mua áo không vừa có đổi được không?"}'
```

Output:
```json
{
    "response": "Chào bạn! Shop hỗ trợ đổi size hoặc đổi mẫu trong vòng 7 ngày kể từ khi nhận hàng..."
}
```
