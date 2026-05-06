from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.routes import router as api_router
from app.services.rag_service import rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components once when server starts."""
    rag_service.initialize()
    yield
    print("🛑 Shutting down RAG Chatbot API...")

# --- FastAPI App ---
app = FastAPI(title="Fashion Store RAG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/favicon.ico")
async def favicon():
    return {"status": "ok"}
