"""
Supervisor Agent - Bộ não điều phối trung tâm.
Phân tích tin nhắn khách hàng và quyết định giao cho Sub-Agent nào xử lý.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import AgentState
import json


# Danh sách các Agent con mà Supervisor có thể điều phối
AGENT_MEMBERS = ["ops_agent", "sales_agent", "support_agent"]

# System Prompt cho Supervisor - CHỈ LÀM NHIỆM VỤ PHÂN LUỒNG
SUPERVISOR_PROMPT = """Bạn là Quản đốc (Supervisor) của hệ thống AI chăm sóc khách hàng cửa hàng thời trang.

## NHIỆM VỤ DUY NHẤT CỦA BẠN:
Đọc tin nhắn của khách hàng và QUYẾT ĐỊNH giao cho ai xử lý. Bạn KHÔNG tự trả lời khách.

## QUY TẮC ƯU TIÊN (ĐỌC TRƯỚC KHI QUYẾT ĐỊNH):
- Nếu khách muốn HỦY đơn hàng (hủy, cancel, bỏ đơn, không muốn mua nữa) → chọn `support_agent` (vì đơn chưa thanh toán sẽ tự động hủy theo chính sách).
- Nếu tin nhắn có chứa BẤT KỲ từ khóa nào liên quan đến chính sách (phí ship, đổi trả, bảo hành, thanh toán, khuyến mãi, chính sách, quy định, hoàn tiền) → LUÔN LUÔN chọn `support_agent`, kể cả khi có nhắc đến sản phẩm.
- Chỉ chọn `sales_agent` khi câu hỏi THUẦN TÚY về sản phẩm/tồn kho (còn hàng, size, màu, giá, SKU) mà KHÔNG có yếu tố chính sách.
- Chỉ chọn `ops_agent` khi câu hỏi về đơn hàng (trạng thái đơn, đơn của tôi, kiểm tra đơn).

## CÁC NHÂN VIÊN DƯỚI QUYỀN:
1. `ops_agent` - Quản lý Đơn hàng: tra cứu trạng thái đơn hàng, kiểm tra đơn, đơn của tôi.
2. `sales_agent` - Tư vấn Bán hàng: sản phẩm, tồn kho, còn hàng, size, màu, giá, SKU, mua, áo, quần, váy, giày.
3. `support_agent` - Hỗ trợ Chính sách: chính sách, đổi trả, bảo hành, thanh toán, khuyến mãi, phí ship, hoàn tiền, quy định, hủy đơn.

## TRƯỜNG HỢP ĐẶC BIỆT:
- Nếu khách chỉ chào hỏi hoặc nói chuyện phiếm -> trả về `FINISH`.
- Nếu câu hỏi không thuộc bất kỳ nhóm nào trên -> trả về `FINISH`.

## CÁCH TRẢ LỜI (BẮT BUỘC):
Trả lời ĐÚNG MỘT TỪ duy nhất. Không giải thích, không thêm dấu chấm, không thêm bất kỳ ký tự nào khác.
ops_agent
sales_agent
support_agent
FINISH"""


def create_supervisor_node():
    """
    Tạo hàm supervisor_node để dùng làm Node trong LangGraph.
    Hàm này đọc state, gọi LLM để phân loại, rồi cập nhật state["next"].
    """
    llm = ChatOllama(model="llama3.1", temperature=0)

    def supervisor_node(state: AgentState) -> AgentState:
        """Node Supervisor: phân tích tin nhắn và quyết định route."""
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            *state["messages"]
        ]

        response = llm.invoke(messages)
        route = response.content.strip().lower().replace('"', '').replace("'", "")

        # Xác định route hợp lệ
        if "cancel" in route:
            next_agent = "support_agent"
        elif "ops" in route:
            next_agent = "ops_agent"
        elif "sale" in route:
            next_agent = "sales_agent"
        elif "support" in route:
            next_agent = "support_agent"
        else:
            next_agent = "FINISH"

        print(f"  🧠 [Supervisor] Routing to: {next_agent}")
        return {"next": next_agent}

    return supervisor_node
