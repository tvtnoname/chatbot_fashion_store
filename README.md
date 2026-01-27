# Fashion Store - AI Consultant Service

Backend service for the Fashion Store AI Consultant.
Powered by Google Gemini and VTO (Virtual Try-On) integration.

## 📁 Structure

```
chatbot_fashion_store/
├── src/fashion_store/    # Backend API (FastAPI/LangChain)
├── requirements.txt      # Python dependencies
└── .env                  # Configuration
```

## 🚀 Installation

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key

### 2. Setup

```bash
cd chatbot_fashion_store

# Create virtual environment
python3 -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_key_here
VTO_API_URL=http://localhost:8000
PORT=5000
```

## ▶️ Running the Service

```bash
# Run with uvicorn
uvicorn src.fashion_store.api:app --host 0.0.0.0 --port 5000 --reload
```

## 📡 API Endpoints

- **POST `/api/chat`**: Chat with the AI consultant.
- **GET `/health`**: Health check.
