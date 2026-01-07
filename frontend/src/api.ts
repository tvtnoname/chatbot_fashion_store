// import axios from 'axios'; // Removed to fix crash

const API_BASE_URL = 'http://localhost:8080';

export interface ImageAnalysisResult {
    category: string;
    color: string;
    material: string;
    style_description: string;
}

export interface ProductRecommendation {
    id: string;
    name: string;
    price: number;
    reason: string;
    image_url?: string;
}

export interface ChatResponse {
    response: string;
    products: ProductRecommendation[];
}

export const api = {
    chat: async (message: string, imageContext?: ImageAnalysisResult) => {
        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    image_context: imageContext
                })
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const data: ChatResponse = await response.json();
            return data;
        } catch (error) {
            console.error("Chat API Error:", error);
            throw error;
        }
    },

    uploadImage: async (file: File) => {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/upload-image`, {
                method: 'POST',
                // Content-Type header is set automatically by fetch for FormData
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload Error: ${response.statusText}`);
            }

            const data: ImageAnalysisResult = await response.json();
            return data;
        } catch (error) {
            console.error("Upload API Error:", error);
            throw error;
        }
    }
};
