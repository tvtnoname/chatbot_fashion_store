import os
import base64
import requests
from typing import Optional
from .vto_dtos import VTORequest, VTOResponse
from dotenv import load_dotenv

load_dotenv()

# VTO API Configuration
VTO_API_URL = os.getenv("VTO_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10000

def call_vto_api(
    user_photo: bytes,
    garment_photo: bytes,
    settings: VTORequest
) -> VTOResponse:
    """
    Call VTO API and return result
    
    Args:
        user_photo: User's photo as bytes
        garment_photo: Garment photo as bytes
        settings: VTO request settings (category, mode, etc.)
        
    Returns:
        VTOResponse with success status and image data or error message
    """
    
    endpoint = f"{VTO_API_URL}/api/vto"
    
    # Prepare multipart files
    files = {
        'model_image': ('user.jpg', user_photo, 'image/jpeg'),
        'garment_image': ('garment.jpg', garment_photo, 'image/jpeg')
    }
    
    # Prepare form data
    data = {
        'category': settings.category,
        'mode': settings.mode,
        'skip_preprocessing': 'true',      # Keep original background
        'preserve_background': 'true',     # Composite on original bg
        'resize_to_input': 'true'          # Match input dimensions
    }
    
    # Headers for ngrok (bypass browser warning)
    headers = {
        'ngrok-skip-browser-warning': 'true'
    }
    
    try:
        # Make request to VTO API
        response = requests.post(
            endpoint,
            files=files,
            data=data,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            # Success - convert PNG bytes to base64
            image_b64 = base64.b64encode(response.content).decode('utf-8')
            
            return VTOResponse(
                success=True,
                image_data=image_b64,
                request_id=response.headers.get('X-Request-ID'),
                processing_time=response.headers.get('X-Processing-Time')
            )
        else:
            # API returned an error
            try:
                error_detail = response.json().get('detail', response.text)
            except:
                error_detail = response.text
                
            return VTOResponse(
                success=False,
                error=f"VTO API error: {error_detail}"
            )
            
    except requests.exceptions.Timeout:
        return VTOResponse(
            success=False,
            error=f"Request timeout - VTO processing took too long (>{REQUEST_TIMEOUT}s). Try with a smaller image or check VTO server status."
        )
    except requests.exceptions.ConnectionError:
        return VTOResponse(
            success=False,
            error=f"Cannot connect to VTO API at {VTO_API_URL}. Please ensure the VTO server is running."
        )
    except Exception as e:
        return VTOResponse(
            success=False,
            error=f"Unexpected error: {str(e)}"
        )
