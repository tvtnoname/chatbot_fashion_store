#!/bin/bash

# Khai báo link Backend trên Render của bạn (Thay <link_render_cua_ban> bằng link thật)
export MAIN_BE_URL="https://fashion-store-api-4wz7.onrender.com/api/v1/internal/chatbot"

# Khởi động ngrok dưới background và ẩn giao diện ngrok đi cho đỡ rối mắt
ngrok http --domain=dischargeable-elin-unexpectantly.ngrok-free.dev 5000 > /dev/null 2>&1 &
NGROK_PID=$!

echo "🚀 Đã khởi động ngrok (PID: $NGROK_PID)"
echo "🚀 Đang khởi động FastAPI (Uvicorn)..."

trap "echo '🛑 Đang tắt ngrok...'; kill $NGROK_PID" EXIT

# Kích hoạt môi trường ảo (virtual environment)
source .venv/bin/activate

# Khởi động uvicorn server
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
