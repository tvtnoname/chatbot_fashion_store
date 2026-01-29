
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from .dtos import ChatRequest, ChatResponse, ImageAnalysisResult, ProductRecommendation
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
        # Run Chatbot Agent
        final_output = run_fashion_consultant(user_query, return_json=False)
        
        return ChatResponse(
            response=str(final_output),
            products=[] 
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
