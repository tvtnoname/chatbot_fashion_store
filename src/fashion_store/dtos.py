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
    id: int
    name: str
    price: float
    similarity_score: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    products: List[ProductRecommendation] = []
