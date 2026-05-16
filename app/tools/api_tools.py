import json
import time
import requests
from typing import Optional, Dict, Any
from langchain_core.tools import tool
import os
from cachetools import TTLCache

# Khởi tạo cache lưu trữ tối đa 100 kết quả trong 180 giây (3 phút)
inventory_cache = TTLCache(maxsize=100, ttl=180)

# Khởi tạo cache cho đơn hàng: TTL ngắn hơn (60 giây) vì trạng thái đơn hàng thay đổi nhanh hơn tồn kho
order_cache = TTLCache(maxsize=50, ttl=60)

MAIN_BE_URL = os.getenv("MAIN_BE_URL", "http://localhost:5001/api/v1/internal/chatbot")

# ── Danh sách từ khóa phổ biến để warm cache khi server khởi động ──
_WARM_QUERIES = ["áo", "quần", "váy", "giày", "phụ kiện"]


def warm_inventory_cache():
    """Pre-fetch các danh mục sản phẩm phổ biến vào cache khi server khởi động.
    Chạy trên background thread để không block server startup."""
    import threading

    def _warm():
        success = 0
        print("🔥 [Cache Warming] Đang tải trước tồn kho phổ biến...")
        for query in _WARM_QUERIES:
            cache_key = (query, None, None)
            if cache_key in inventory_cache:
                continue  # Đã có trong cache (trường hợp reload)
            try:
                params = {"q": query}
                res = requests.get(f"{MAIN_BE_URL}/inventory", params=params, timeout=15)
                data = res.json()
                if data.get("status") == "success":
                    results = data.get("data", [])
                    if results:
                        output = []
                        for item in results:
                            output.append(
                                f"Sản phẩm: {item['product_name']} (SKU: {item['sku']}), "
                                f"Size: {item['size']}, Màu: {item['color']}, "
                                f"Tồn kho: {item['stock_qty']}, Giá: {item['price']}đ"
                            )
                        result = json.dumps(
                            {"text_summary": "\n".join(output), "raw_products": results},
                            ensure_ascii=False,
                        )
                    else:
                        result = json.dumps(
                            {"text_summary": f"Không tìm thấy tồn kho cho: {query}", "raw_products": []},
                            ensure_ascii=False,
                        )
                    inventory_cache[cache_key] = result
                    success += 1
                    print(f"    ✅ [Warm] '{query}' → {len(results)} sản phẩm cached")
                else:
                    print(f"    ⚠️ [Warm] '{query}' → API trả lỗi: {data.get('message')}")
            except Exception as e:
                print(f"    ❌ [Warm] '{query}' → Lỗi: {e}")
        print(f"🔥 [Cache Warming] Hoàn tất! {success}/{len(_WARM_QUERIES)} danh mục đã cached.")

    # Chạy trên background thread để không chặn server startup
    thread = threading.Thread(target=_warm, daemon=True)
    thread.start()

def warm_order_cache(user_id: str):
    """Pre-fetch đơn hàng của một user cụ thể vào cache.
    Vì dữ liệu đơn hàng gắn với user, ta không thể warm lúc startup.
    Thay vào đó, gọi hàm này chạy ngầm ngay khi user gửi tin nhắn đến chatbot."""
    if not user_id or str(user_id) == "Chưa đăng nhập":
        return

    import threading
    def _warm():
        try:
            uid = int(user_id)
            if uid <= 0:
                return
            
            cache_key = (uid, None)
            if cache_key in order_cache:
                return  # Đã có cache gần đây
                
            print(f"    🔥 [Order Warm] Đang tải trước đơn hàng cho user {uid}...")
            params = {"user_id": uid}
            res = requests.get(f"{MAIN_BE_URL}/orders", params=params, timeout=10)
            data = res.json()
            
            if data.get("status") == "success":
                order = data.get("data", {})
                result = f"Đơn hàng #{order.get('order_id')} tạo lúc {order.get('created_at')}. Trạng thái hiện tại: {order.get('status')}. Tổng tiền: {order.get('total_amount')}đ."
            else:
                result = data.get("message", "Không tìm thấy đơn hàng.")
                
            order_cache[cache_key] = result
            print(f"    ✅ [Order Warm] Đã cache đơn hàng cho user {uid}")
        except Exception as e:
            print(f"    ❌ [Order Warm] Lỗi khi tải trước đơn hàng user {user_id}: {e}")

    # Chạy trên background thread để không chặn luồng chat chính
    thread = threading.Thread(target=_warm, daemon=True)
    thread.start()

@tool
def check_inventory(query: str, size: Optional[str] = None, color: Optional[str] = None) -> str:
    """Tra cứu tồn kho thực tế của sản phẩm dựa vào tên hoặc SKU, size, color."""
    # ── Cache thủ công (tương thích với @tool decorator của LangChain) ──
    cache_key = (query, size, color)
    if cache_key in inventory_cache:
        print(f"    ⚡ [Cache HIT] inventory: {cache_key}")
        return inventory_cache[cache_key]
    print(f"    🌐 [Cache MISS] inventory: {cache_key} → gọi API...")

    params = {"q": query}
    if size:
        params["size"] = size
    if color:
        params["color"] = color
        
    try:
        t0 = time.time()
        print(f"    📡 [API] Requesting: {MAIN_BE_URL}/inventory?{params}")
        res = requests.get(f"{MAIN_BE_URL}/inventory", params=params, timeout=10)
        print(f"    ✅ [API] Response received in {time.time()-t0:.2f}s (status={res.status_code})")
        data = res.json()
        if data.get("status") == "success":
            results = data.get("data", [])
            if not results:
                result = json.dumps({"text_summary": f"Không tìm thấy tồn kho cho: {query}", "raw_products": []}, ensure_ascii=False)
            else:
                output = []
                for item in results:
                    output.append(f"Sản phẩm: {item['product_name']} (SKU: {item['sku']}), Size: {item['size']}, Màu: {item['color']}, Tồn kho: {item['stock_qty']}, Giá: {item['price']}đ")
                result = json.dumps({"text_summary": "\n".join(output), "raw_products": results}, ensure_ascii=False)
        else:
            result = json.dumps({"text_summary": data.get("message", "Có lỗi khi tra cứu tồn kho."), "raw_products": []}, ensure_ascii=False)
    except Exception as e:
        print(f"    ❌ [API] Error after {time.time()-t0:.2f}s: {e}")
        result = json.dumps({"text_summary": f"Lỗi kết nối tra cứu tồn kho: {str(e)}", "raw_products": []}, ensure_ascii=False)
    
    print(f"    📦 [Tool] check_inventory done. Result length: {len(result)} chars")
    inventory_cache[cache_key] = result
    return result

@tool
def check_order_status(user_id: str, order_id: Optional[str] = None) -> str:
    """Tra cứu trạng thái đơn hàng của khách hàng. Yêu cầu truyền đúng user_id (số nguyên). Truyền order_id nếu khách muốn hỏi một đơn cụ thể."""
    # ── Validate & convert user_id ──
    try:
        uid = int(user_id)
    except (ValueError, TypeError):
        print(f"    ⚠️ [Tool] Invalid user_id: {user_id}")
        return "Khách hàng chưa đăng nhập. Vui lòng đăng nhập để kiểm tra đơn hàng."

    if uid <= 0:
        return "Khách hàng chưa đăng nhập. Vui lòng đăng nhập để kiểm tra đơn hàng."

    # ── Validate & convert order_id (nếu có) ──
    oid = None
    if order_id:
        try:
            oid = int(order_id)
        except (ValueError, TypeError):
            oid = None

    # ── Cache thủ công ──
    cache_key = (uid, oid)
    if cache_key in order_cache:
        print(f"    ⚡ [Cache HIT] order: {cache_key}")
        return order_cache[cache_key]
    print(f"    🌐 [Cache MISS] order: {cache_key} → gọi API...")

    params = {"user_id": uid}
    if oid:
        params["order_id"] = oid
        
    try:
        t0 = time.time()
        print(f"    📡 [API] Requesting: {MAIN_BE_URL}/orders?{params}")
        res = requests.get(f"{MAIN_BE_URL}/orders", params=params, timeout=10)
        print(f"    ✅ [API] Response received in {time.time()-t0:.2f}s (status={res.status_code})")
        data = res.json()
        if data.get("status") == "success":
            order = data.get("data", {})
            result = f"Đơn hàng #{order.get('order_id')} tạo lúc {order.get('created_at')}. Trạng thái hiện tại: {order.get('status')}. Tổng tiền: {order.get('total_amount')}đ."
        else:
            result = data.get("message", "Không tìm thấy đơn hàng.")
    except Exception as e:
        print(f"    ❌ [API] Error after {time.time()-t0:.2f}s: {e}")
        result = f"Lỗi kết nối tra cứu đơn hàng: {str(e)}"
    
    print(f"    📦 [Tool] check_order_status done. Result length: {len(result)} chars")
    order_cache[cache_key] = result
    return result
