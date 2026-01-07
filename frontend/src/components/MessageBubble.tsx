// import ReactMarkdown from 'react-markdown'; // Unused for now
import ProductCard from './ProductCard';
import { User, Sparkles } from 'lucide-react';

interface MessageBubbleProps {
    message: {
        role: 'user' | 'assistant';
        content: string;
        image?: string;
        products?: any[];
    };
}

export default function MessageBubble({ message }: MessageBubbleProps) {
    const isUser = message.role === 'user';

    return (
        <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
            {/* Avatar for Bot */}
            {!isUser && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-white" />
                </div>
            )}

            <div className={`max-w-[80%] rounded-2xl p-4 ${isUser ? 'bg-indigo-600 text-white' : 'bg-white shadow-sm border text-gray-800'
                }`}>
                {message.image && (
                    <img src={message.image} alt="Uploaded" className="mb-2 max-w-xs rounded-lg border-2 border-white/20" />
                )}
                <div className="prose prose-sm max-w-none">
                    {/* Temporary simple text display, ideally Markdown */}
                    <p className="whitespace-pre-wrap">{message.content}</p>
                </div>

                {/* Product Recommendations */}
                {message.products && message.products.length > 0 && (
                    <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 overflow-x-auto">
                        {message.products.map((p, idx) => (
                            <ProductCard key={idx} product={p} />
                        ))}
                    </div>
                )}
            </div>

            {/* Avatar for User */}
            {isUser && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                    <User className="w-5 h-5 text-gray-600" />
                </div>
            )}
        </div>
    );
}
