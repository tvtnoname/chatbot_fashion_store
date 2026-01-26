import { useState } from 'react';
import { X, Upload, Loader2, Download, Sparkles } from 'lucide-react';
import { api, type VTOResponse } from '../api';

interface VirtualTryOnProps {
    onClose: () => void;
}

export default function VirtualTryOn({ onClose }: VirtualTryOnProps) {
    const [userPhoto, setUserPhoto] = useState<File | null>(null);
    const [garmentPhoto, setGarmentPhoto] = useState<File | null>(null);
    const [userPhotoPreview, setUserPhotoPreview] = useState<string | null>(null);
    const [garmentPhotoPreview, setGarmentPhotoPreview] = useState<string | null>(null);
    const [category, setCategory] = useState<'upperbody' | 'lowerbody' | 'dress'>('upperbody');
    const [mode, setMode] = useState<'hd' | 'dc'>('hd');
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState<VTOResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFileSelect = (file: File, type: 'user' | 'garment') => {
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return;
        }

        const reader = new FileReader();
        reader.onloadend = () => {
            if (type === 'user') {
                setUserPhoto(file);
                setUserPhotoPreview(reader.result as string);
            } else {
                setGarmentPhoto(file);
                setGarmentPhotoPreview(reader.result as string);
            }
        };
        reader.readAsDataURL(file);
    };

    const handleTryOn = async () => {
        if (!userPhoto || !garmentPhoto) {
            alert('Please upload both photos');
            return;
        }

        setIsProcessing(true);
        setError(null);
        setResult(null);

        try {
            const response = await api.virtualTryOn({
                userPhoto,
                garmentPhoto,
                category,
                mode
            });

            if (response.success) {
                setResult(response);
            } else {
                setError(response.error || 'Try-on failed');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleDownload = () => {
        if (!result?.image_data) return;

        const link = document.createElement('a');
        link.href = `data:image/png;base64,${result.image_data}`;
        link.download = `try-on-result-${Date.now()}.png`;
        link.click();
    };

    const handleReset = () => {
        setResult(null);
        setError(null);
        setUserPhoto(null);
        setGarmentPhoto(null);
        setUserPhotoPreview(null);
        setGarmentPhotoPreview(null);
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b sticky top-0 bg-white z-10">
                    <div className="flex items-center gap-3">
                        <Sparkles className="w-6 h-6 text-purple-600" />
                        <h2 className="text-2xl font-bold text-gray-800">Virtual Try-On</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700 transition-colors"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6">
                    {!result ? (
                        <>
                            {/* Upload Section */}
                            <div className="grid md:grid-cols-2 gap-6 mb-6">
                                {/* User Photo Upload */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                                        Your Photo 📸
                                    </label>
                                    <div
                                        className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center hover:border-purple-500 transition-colors cursor-pointer"
                                        onClick={() => document.getElementById('user-photo-input')?.click()}
                                    >
                                        {userPhotoPreview ? (
                                            <img
                                                src={userPhotoPreview}
                                                alt="User preview"
                                                className="w-full h-64 object-cover rounded-lg mb-2"
                                            />
                                        ) : (
                                            <div className="flex flex-col items-center justify-center h-64">
                                                <Upload className="w-12 h-12 text-gray-400 mb-2" />
                                                <p className="text-gray-600 text-sm">Click to upload your photo</p>
                                                <p className="text-gray-400 text-xs mt-1">Max 10MB</p>
                                            </div>
                                        )}
                                        <input
                                            id="user-photo-input"
                                            type="file"
                                            accept="image/*"
                                            className="hidden"
                                            onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0], 'user')}
                                        />
                                    </div>
                                </div>

                                {/* Garment Photo Upload */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                                        Garment Photo 👔
                                    </label>
                                    <div
                                        className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center hover:border-purple-500 transition-colors cursor-pointer"
                                        onClick={() => document.getElementById('garment-photo-input')?.click()}
                                    >
                                        {garmentPhotoPreview ? (
                                            <img
                                                src={garmentPhotoPreview}
                                                alt="Garment preview"
                                                className="w-full h-64 object-cover rounded-lg mb-2"
                                            />
                                        ) : (
                                            <div className="flex flex-col items-center justify-center h-64">
                                                <Upload className="w-12 h-12 text-gray-400 mb-2" />
                                                <p className="text-gray-600 text-sm">Click to upload garment photo</p>
                                                <p className="text-gray-400 text-xs mt-1">Max 10MB</p>
                                            </div>
                                        )}
                                        <input
                                            id="garment-photo-input"
                                            type="file"
                                            accept="image/*"
                                            className="hidden"
                                            onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0], 'garment')}
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* Settings */}
                            <div className="grid md:grid-cols-2 gap-6 mb-6">
                                {/* Category */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                                        Category
                                    </label>
                                    <select
                                        value={category}
                                        onChange={(e) => setCategory(e.target.value as any)}
                                        className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                    >
                                        <option value="upperbody">Upper Body (Shirts, Jackets)</option>
                                        <option value="lowerbody">Lower Body (Pants, Skirts)</option>
                                        <option value="dress">Dress</option>
                                    </select>
                                </div>

                                {/* Mode */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                                        Quality Mode
                                    </label>
                                    <select
                                        value={mode}
                                        onChange={(e) => setMode(e.target.value as any)}
                                        className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                    >
                                        <option value="hd">HD (High Quality, Slower)</option>
                                        <option value="dc">DC (Draft, Faster)</option>
                                    </select>
                                </div>
                            </div>

                            {/* Error Display */}
                            {error && (
                                <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                                    <p className="text-red-700 text-sm">{error}</p>
                                </div>
                            )}

                            {/* Try On Button */}
                            <button
                                onClick={handleTryOn}
                                disabled={!userPhoto || !garmentPhoto || isProcessing}
                                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold py-4 rounded-xl hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] flex items-center justify-center gap-2"
                            >
                                {isProcessing ? (
                                    <>
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                        Processing... (10-15 seconds)
                                    </>
                                ) : (
                                    <>
                                        <Sparkles className="w-5 h-5" />
                                        Try On Now
                                    </>
                                )}
                            </button>
                        </>
                    ) : (
                        <>
                            {/* Result Display */}
                            <div className="mb-6">
                                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                                    <Sparkles className="w-5 h-5 text-purple-600" />
                                    Try-On Result
                                </h3>

                                <div className="grid md:grid-cols-2 gap-4 mb-4">
                                    {/* Before */}
                                    <div>
                                        <p className="text-sm font-semibold text-gray-600 mb-2">Before</p>
                                        <img
                                            src={userPhotoPreview || ''}
                                            alt="Before"
                                            className="w-full rounded-lg border-2 border-gray-200"
                                        />
                                    </div>

                                    {/* After */}
                                    <div>
                                        <p className="text-sm font-semibold text-gray-600 mb-2">After</p>
                                        <img
                                            src={`data:image/png;base64,${result.image_data}`}
                                            alt="Try-on result"
                                            className="w-full rounded-lg border-2 border-purple-500 shadow-lg"
                                        />
                                    </div>
                                </div>

                                {/* Processing Info */}
                                {result.processing_time && (
                                    <p className="text-sm text-gray-500 text-center mb-4">
                                        Processing time: {result.processing_time}
                                    </p>
                                )}

                                {/* Action Buttons */}
                                <div className="flex gap-4">
                                    <button
                                        onClick={handleDownload}
                                        className="flex-1 bg-green-600 text-white font-semibold py-3 rounded-xl hover:bg-green-700 transition-colors flex items-center justify-center gap-2"
                                    >
                                        <Download className="w-5 h-5" />
                                        Download Result
                                    </button>
                                    <button
                                        onClick={handleReset}
                                        className="flex-1 bg-gray-200 text-gray-700 font-semibold py-3 rounded-xl hover:bg-gray-300 transition-colors"
                                    >
                                        Try Another
                                    </button>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
