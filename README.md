# Fashion Store - AI Virtual Try-On

E-commerce fashion store with AI-powered virtual try-on and fashion consultant.

## Structure

```
fashion_store/
├── src/fashion_store/    # Backend API
├── frontend/             # React frontend
├── scripts/              # Helper scripts
└── requirements.txt      # Python dependencies
```

## Quick Start

### Backend (WSL Terminal)

```bash
cd /mnt/e/AI_Agent/fashion_store/

# Create virtual environment
python3 -m venv ~/fashion_env
source ~/fashion_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn src.fashion_store.api:app --port 5000 --reload
```

**API**: http://localhost:5000  
**Docs**: http://localhost:5000/docs

### Frontend (Windows Terminal)

```bash  
cd E:\AI_Agent\fashion_store\frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

**Frontend**: http://localhost:3000

## Features

- 🤖 AI Fashion Consultant (Gemini-powered agents)
- 👔 Virtual Try-On integration
- 🛍️ Product catalog
- 💬 Chat-based recommendations

## API Endpoints

```
POST /api/virtual-try-on  - VTO endpoint
POST /api/chat            - AI consultant
GET  /api/products        - Product list
GET  /health              - Health check
```

## Configuration

Edit `.env`:
```bash
VTO_API_URL=http://localhost:8000
GEMINI_API_KEY=your_key_here
```

## Architecture

```
Frontend (3000) → Backend API (5000) → VTO API (8000) → OOTDiffusion (7865)
```

## Development

**Backend hot reload**: Uvicorn auto-reloads on file changes  
**Frontend hot reload**: Vite HMR enabled

## Testing

```bash
# Test VTO endpoint
python test_vto_ngrok.py user.jpg garment.jpg
```
