import json
import requests
from typing import Optional, Dict, Any
from langchain_core.tools import tool
import os
from cachetools import cached, TTLCache

# Khởi tạo cache lưu trữ tối đa 100 kết quả trong 180 giây (3 phút)
inventory_cache = TTLCache(maxsize=100, ttl=180)

# Khởi tạo cache cho đơn hàng: TTL ngắn hơn (60 giây) vì trạng thái đơn hàng thay đổi nhanh hơn tồn kho
order_cache = TTLCache(maxsize=50, ttl=60)

MAIN_BE_URL = os.getenv("MAIN_BE_URL", "http://localhost:5001/api/v1/internal/chatbot")

@tool
@cached(cache=inventory_cache)
def check_inventory(query: str, size: Optional[str] = None, color: Optional[str] = None) -> str:
    """Tra cứu tồn kho thực tế của sản phẩm dựa vào tên hoặc SKU, size, color."""
    params = {"q": query}
    if size:
        params["size"] = size
    if color:
        params["color"] = color
        
    try:
        res = requests.get(f"{MAIN_BE_URL}/inventory", params=params, timeout=10)
        data = res.json()
        if data.get("status") == "success":
            results = data.get("data", [])
            if not results:
                return json.dumps({"text_summary": f"Không tìm thấy tồn kho cho: {query}", "raw_products": []}, ensure_ascii=False)
            
            output = []
            for item in results:
                output.append(f"Sản phẩm: {item['product_name']} (SKU: {item['sku']}), Size: {item['size']}, Màu: {item['color']}, Tồn kho: {item['stock_qty']}, Giá: {item['price']}đ")
            return json.dumps({"text_summary": "\n".join(output), "raw_products": results}, ensure_ascii=False)
        else:
            return json.dumps({"text_summary": data.get("message", "Có lỗi khi tra cứu tồn kho."), "raw_products": []}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"text_summary": f"Lỗi kết nối tra cứu tồn kho: {str(e)}", "raw_products": []}, ensure_ascii=False)

@tool
@cached(cache=order_cache)
def check_order_status(user_id: int, order_id: Optional[int] = None) -> str:
    """Tra cứu trạng thái đơn hàng của khách hàng. Yêu cầu truyền đúng user_id. Truyền order_id nếu khách muốn hỏi một đơn cụ thể."""
    if not user_id:
        return "Yêu cầu đăng nhập để kiểm tra đơn hàng."
        
    params = {"user_id": user_id}
    if order_id:
        params["order_id"] = order_id
        
    try:
        res = requests.get(f"{MAIN_BE_URL}/orders", params=params, timeout=10)
        data = res.json()
        if data.get("status") == "success":
            order = data.get("data", {})
            return f"Đơn hàng #{order.get('order_id')} tạo lúc {order.get('created_at')}. Trạng thái hiện tại: {order.get('status')}. Tổng tiền: {order.get('total_amount')}đ."
        else:
            return data.get("message", "Không tìm thấy đơn hàng.")
    except Exception as e:
        return f"Lỗi kết nối tra cứu đơn hàng: {str(e)}"

