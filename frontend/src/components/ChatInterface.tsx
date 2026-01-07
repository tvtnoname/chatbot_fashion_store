import { useState } from 'react';
import { Send, Loader2, ShoppingBag } from 'lucide-react';
import { api, type ChatResponse, type ImageAnalysisResult } from '../api';
import MessageBubble from './MessageBubble';
import ImageUpload from './ImageUpload';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    image?: string; // Base64 or URL
    products?: any[];
}

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [imageContext, setImageContext] = useState<ImageAnalysisResult | undefined>(undefined);
    const [previewImage, setPreviewImage] = useState<string | null>(null);

    const handleSendMessage = async () => {
        if (!inputText.trim() && !imageContext) return;

        const newMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: inputText,
            image: previewImage || undefined
        };

        setMessages(prev => [...prev, newMessage]);
        setIsLoading(true);
        setInputText('');

        try {
            const response = await api.chat(newMessage.content, imageContext);

            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: response.response,
                products: response.products
            };

            setMessages(prev => [...prev, botMessage]);
            // Reset context after sending
            setImageContext(undefined);
            setPreviewImage(null);
        } catch (error) {
            console.error(error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: "Xin lỗi, tôi gặp sự cố khi kết nối với Agent. Vui lòng thử lại."
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleImageUpload = async (file: File) => {
        setIsLoading(true);
        try {
            // Preview
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewImage(reader.result as string);
            };
            reader.readAsDataURL(file);

            // Analyze
            const result = await api.uploadImage(file);
            setImageContext(result);

            // Auto add system message about generic analysis
            const analysisMsg: Message = {
                id: Date.now().toString(),
                role: 'assistant',
                content: `Tôi đã nhận diện được sản phẩm: ${result.category} màu ${result.color}. Bạn muốn tôi tư vấn gì về món này?`
            };
            setMessages(prev => [...prev, analysisMsg]);

        } catch (error) {
            console.error(error);
            alert("Lỗi tải ảnh!");
            setPreviewImage(null);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white shadow-sm p-4 flex items-center justify-between">
                <h1 className="text-xl font-bold flex items-center gap-2 text-indigo-600">
                    <ShoppingBag className="w-6 h-6" />
                    Fashion AI Stylist
                </h1>
            </header>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map(msg => (
                    <MessageBubble key={msg.id} message={msg} />
                ))}
                {isLoading && (
                    <div className="flex items-center gap-2 text-gray-500 text-sm p-4">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        AI Stylist đang suy nghĩ...
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white border-t">
                {previewImage && (
                    <div className="mb-2 relative w-20 h-20">
                        <img src={previewImage} alt="Preview" className="w-full h-full object-cover rounded-md border" />
                        <button onClick={() => { setPreviewImage(null); setImageContext(undefined); }} className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-5 h-5 text-xs">x</button>
                    </div>
                )}
                <div className="flex gap-2">
                    <ImageUpload onUpload={handleImageUpload} />
                    <input
                        type="text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                        placeholder="Hỏi tôi về cách phối đồ..."
                        className="flex-1 border rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={isLoading || (!inputText && !imageContext)}
                        className="bg-indigo-600 text-white p-2 rounded-full hover:bg-indigo-700 disabled:opacity-50"
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </div>
            </div>
        </div>
    );
}
