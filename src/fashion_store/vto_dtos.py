from pydantic import BaseModel, Field
from typing import Optional


class VTORequest(BaseModel):
    """Virtual Try-On request parameters"""
    category: str = Field(
        default="upperbody",
        description="Garment category: upperbody, lowerbody, or dress"
    )
    mode: str = Field(
        default="hd",
        description="Processing mode: hd (high definition) or dc (draft copy)"
    )
    skip_preprocessing: bool = Field(
        default=False,
        description="Skip background removal preprocessing"
    )


class VTOResponse(BaseModel):
    """Virtual Try-On response"""
    success: bool
    image_data: Optional[str] = None  # Base64 encoded PNG
    error: Optional[str] = None
    request_id: Optional[str] = None
    processing_time: Optional[str] = None
