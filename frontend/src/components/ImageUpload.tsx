import React, { useRef } from 'react';
import { Camera } from 'lucide-react';

interface ImageUploadProps {
    onUpload: (file: File) => void;
}

export default function ImageUpload({ onUpload }: ImageUploadProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const checkFile = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            onUpload(e.target.files[0]);
        }
    };

    return (
        <>
            <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={checkFile}
            />
            <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 text-gray-500 hover:text-indigo-600 hover:bg-gray-100 rounded-full transition-colors"
                title="Tải ảnh lên"
            >
                <Camera className="w-6 h-6" />
            </button>
        </>
    );
}
