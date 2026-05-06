from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import rag_service

router = APIRouter()

@router.get("/")
async def root():
    return {"status": "ok", "message": "Fashion Store RAG Chatbot API is running (Ollama + ChromaDB)"}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        question = request.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
            
        response_text = rag_service.get_answer(question)
        return ChatResponse(response=response_text)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
