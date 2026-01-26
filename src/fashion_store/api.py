
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from .dtos import ChatRequest, ChatResponse, ImageAnalysisResult, ProductRecommendation
from .vto_dtos import VTOResponse, VTORequest
from .vto_service import call_vto_api
from .main import run_fashion_consultant
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import json
import base64

app = FastAPI(title="Fashion Stylist Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini for Vision
vision_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

@app.post("/upload-image", response_model=ImageAnalysisResult)
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Encode to base64 for LangChain/Gemini
        image_b64 = base64.b64encode(contents).decode("utf-8")
        
        prompt = (
            "Analyze this fashion image and extract the following details in JSON format:\n"
            "- category: The type of item (e.g., Shirt, Dress, Pants)\n"
            "- color: Main color\n"
            "- material: Likely material (e.g., Cotton, Denim, Silk)\n"
            "- style_description: A brief description of the style/vibe (e.g., Minimalist, Streetwear, Elegant)\n"
            "Output ONLY raw JSON."
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        )
        
        response = vision_llm.invoke([message])
        content = response.content.replace('```json', '').replace('```', '').strip()
        data = json.loads(content)
        
        return ImageAnalysisResult(**data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Construct complete prompt
    user_query = request.message
    if request.image_context:
        ctx = request.image_context
        user_query += f"\n\n[CONTEXT TỪ ẢNH]: Người dùng đang hỏi về một sản phẩm có đặc điểm: {ctx.category} màu {ctx.color}, chất liệu {ctx.material}, phong cách {ctx.style_description}. Hãy tư vấn cách phối đồ với món này."
    
    try:
        # Run CrewAI
        # Note: This is synchronous/blocking for now. Ideally should be async or background task.
        # But CrewAI is mostly sync. We wrap in a simple function call.
        
        final_output = run_fashion_consultant(user_query, return_json=False)
        
        # Simple parsing logic (Ideal: Agent return structured JSON)
        # For now, we return the whole text.
        # TODO: Refactor Agent to return structured output for Product Cards
        
        return ChatResponse(
            response=str(final_output),
            products=[] # Populating this requires stricter Agent output format
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/virtual-try-on", response_model=VTOResponse)
async def virtual_try_on(
    user_photo: UploadFile = File(..., description="User's photo (max 10MB)"),
    garment_photo: UploadFile = File(..., description="Garment photo (max 10MB)"),
    category: str = Form("upperbody", description="upperbody, lowerbody, or dress"),
    mode: str = Form("hd", description="hd or dc")
):
    """
    Virtual Try-On endpoint
    
    Upload user photo and garment photo to generate a try-on result.
    Processing time: 10-15 seconds.
    """
    
    # File size validation (10MB max)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    
    try:
        # Read user photo
        user_photo_bytes = await user_photo.read()
        if len(user_photo_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"User photo too large. Maximum size is 10MB."
            )
        
        # Read garment photo
        garment_photo_bytes = await garment_photo.read()
        if len(garment_photo_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Garment photo too large. Maximum size is 10MB."
            )
        
        # Validate category
        valid_categories = ["upperbody", "lowerbody", "dress"]
        if category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )
        
        # Validate mode
        valid_modes = ["hd", "dc"]
        if mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode. Must be one of: {', '.join(valid_modes)}"
            )
        
        # Create request settings
        settings = VTORequest(
            category=category,
            mode=mode,
            skip_preprocessing=False
        )
        
        # Call VTO service
        result = call_vto_api(
            user_photo=user_photo_bytes,
            garment_photo=garment_photo_bytes,
            settings=settings
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Virtual try-on processing failed: {str(e)}"
        )
