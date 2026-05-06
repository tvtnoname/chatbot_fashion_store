import requests
from typing import Optional, Dict, Any
from langchain_core.tools import tool

MAIN_BE_URL = "http://localhost:5001/api/v1/internal/chatbot"

@tool
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
                return f"Không tìm thấy tồn kho cho: {query}"
            
            output = []
            for item in results:
                output.append(f"Sản phẩm: {item['product_name']} (SKU: {item['sku']}), Size: {item['size']}, Màu: {item['color']}, Tồn kho: {item['stock_qty']}, Giá: {item['price']}đ")
            return "\n".join(output)
        else:
            return data.get("message", "Có lỗi khi tra cứu tồn kho.")
    except Exception as e:
        return f"Lỗi kết nối tra cứu tồn kho: {str(e)}"

@tool
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
            return f"Đơn hàng #{order.get('order_id')} tạo lúc {order.get('created_at')}. Trạng thái hiện tại: {order.get('status')}. Tổng tiền: {order.get('total_amount')}đ. Mã vận đơn: {order.get('tracking_number') or 'Chưa có'}."
        else:
            return data.get("message", "Không tìm thấy đơn hàng.")
    except Exception as e:
        return f"Lỗi kết nối tra cứu đơn hàng: {str(e)}"

@tool
def cancel_order(order_id: int, user_id: int) -> str:
    """Hủy đơn hàng nếu khách hàng yêu cầu. Yêu cầu phải có order_id và user_id của khách."""
    if not user_id:
        return "Yêu cầu đăng nhập để thực hiện hủy đơn."
        
    try:
        res = requests.post(f"{MAIN_BE_URL}/orders/{order_id}/cancel", json={"user_id": user_id}, timeout=10)
        data = res.json()
        return data.get("message", "Có lỗi xảy ra khi hủy đơn.")
    except Exception as e:
        return f"Lỗi kết nối hủy đơn: {str(e)}"
