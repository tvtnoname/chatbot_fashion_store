

interface ProductCardProps {
    product: {
        name: string;
        price: number;
        reason: string;
        image_url?: string;
    }
}

export default function ProductCard({ product }: ProductCardProps) {
    return (
        <div className="bg-gray-50 rounded-lg p-3 border hover:shadow-md transition-shadow">
            {product.image_url && (
                <div className="w-full h-32 bg-gray-200 rounded-md mb-2 overflow-hidden">
                    <img src={product.image_url} alt={product.name} className="w-full h-full object-cover" />
                </div>
            )}
            <h4 className="font-semibold text-sm text-gray-900 line-clamp-1">{product.name}</h4>
            <p className="text-indigo-600 font-bold text-sm">${product.price}</p>
            <p className="text-xs text-gray-500 mt-1 italic">"{product.reason}"</p>
        </div>
    );
}
