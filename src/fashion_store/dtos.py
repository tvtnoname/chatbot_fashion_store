from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ImageAnalysisResult(BaseModel):
    category: str
    color: str
    material: str
    style_description: str

class ChatRequest(BaseModel):
    message: str
    image_context: Optional[ImageAnalysisResult] = None

class ProductRecommendation(BaseModel):
    id: str
    name: str
    price: float
    reason: str
    image_url: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    products: List[ProductRecommendation] = []
