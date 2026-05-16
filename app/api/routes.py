from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import rag_service
from app.tools.api_tools import warm_order_cache

router = APIRouter()

@router.get("/")
async def root():
    return {"status": "ok", "message": "Fashion Store RAG Chatbot API is running (Ollama + ChromaDB)"}

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        question = request.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
            
        # ── Trigger Order Cache Warming ngầm ──
        if request.user_id:
            warm_order_cache(str(request.user_id))
            
        result = rag_service.get_answer(question, request.user_id, request.thread_id)
        return ChatResponse(
            response=result["response"],
            thread_id=result.get("thread_id", ""),
            products=result.get("products", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    try:
        question = request.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
            
        # ── Trigger Order Cache Warming ngầm ──
        if request.user_id:
            warm_order_cache(str(request.user_id))
            
        generator = rag_service.astream_answer(question, request.user_id, request.thread_id)
        return StreamingResponse(generator, media_type="text/event-stream")
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
