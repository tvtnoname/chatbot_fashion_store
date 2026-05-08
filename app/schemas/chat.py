from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str = ""
    products: Optional[list] = []
