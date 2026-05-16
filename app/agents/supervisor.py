"""
Supervisor Agent - Bộ não điều phối trung tâm.
Phân tích tin nhắn khách hàng, bóc tách đa ý định (Intent Splitting),
và quyết định giao cho một hoặc nhiều Sub-Agent xử lý tuần tự.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import List
from app.agents.state import AgentState
import json


# Danh sách các Agent con mà Supervisor có thể điều phối
AGENT_MEMBERS = ["ops_agent", "sales_agent", "support_agent"]


# ── Pydantic Schema cho Structured Output ──
class SupervisorOutput(BaseModel):
    """Schema JSON mà Supervisor phải trả về."""
    agents: List[str] = Field(
        default_factory=list,
        description="Danh sách tên agent cần gọi. Để trống nếu là chào hỏi."
    )
    direct_response: str = Field(
        default="",
        description="Câu trả lời trực tiếp bằng tiếng Việt nếu là chào hỏi hoặc phiếm luận. Để trống nếu cần gọi agent."
    )


# System Prompt cho Supervisor - HỖ TRỢ ĐA Ý ĐỊNH
SUPERVISOR_PROMPT = """Bạn là Quản đốc (Supervisor) của hệ thống AI chăm sóc khách hàng cửa hàng thời trang.

## NHIỆM VỤ DUY NHẤT CỦA BẠN:
Đọc tin nhắn của khách hàng, PHÂN TÍCH tất cả các ý định trong đó, và quyết định giao cho agent nào xử lý.

## PHÂN TÍCH ĐA Ý ĐỊNH (RẤT QUAN TRỌNG):
- Một tin nhắn có thể chứa NHIỀU ý định. Ví dụ: "Đơn #32 trạng thái gì và lỗi có đổi trả không?" → 2 ý: đơn hàng (ops_agent) VÀ chính sách (support_agent).
- Hãy liệt kê TẤT CẢ các agent cần thiết vào mảng "agents".

## CÁC NHÂN VIÊN DƯỚI QUYỀN:
1. `ops_agent` - Quản lý Đơn hàng: tra cứu trạng thái đơn hàng, kiểm tra đơn, đơn của tôi.
2. `sales_agent` - Tư vấn Bán hàng: sản phẩm, tồn kho, còn hàng, size, màu, giá, SKU, mua, áo, quần, váy, giày.
3. `support_agent` - Hỗ trợ Chính sách: chính sách, đổi trả, bảo hành, thanh toán, khuyến mãi, phí ship, hoàn tiền, quy định, hủy đơn.

## QUY TẮC ƯU TIÊN:
- Nếu khách muốn HỦY đơn hàng → chọn support_agent.
- Nếu tin nhắn có cả yếu tố đơn hàng VÀ chính sách → chọn CẢ HAI: ops_agent và support_agent.
- Nếu tin nhắn có cả yếu tố sản phẩm VÀ chính sách → chọn CẢ HAI: sales_agent và support_agent.

## TRƯỜNG HỢP CHÀO HỎI / PHIẾM LUẬN:
- Nếu khách chỉ chào hỏi, cảm ơn, hoặc nói chuyện phiếm → KHÔNG chọn agent nào.
- Thay vào đó, viết câu trả lời thân thiện bằng tiếng Việt vào "direct_response" (2-3 câu, gợi ý khách hỏi về sản phẩm/đơn hàng/chính sách).

## CÁCH TRẢ LỜI (BẮT BUỘC THEO JSON):
Trả về JSON với 2 field:
- "agents": mảng tên agent (ví dụ: ["ops_agent", "support_agent"]). Rỗng nếu chào hỏi.
- "direct_response": câu trả lời trực tiếp bằng tiếng Việt. Rỗng nếu cần gọi agent."""


def create_supervisor_node():
    """
    Tạo hàm supervisor_node để dùng làm Node trong LangGraph.
    Sử dụng Structured Output với fallback regex parsing.
    """
    llm = ChatOllama(model="llama3.1", temperature=0)
    structured_llm = llm.with_structured_output(SupervisorOutput, method="json_schema")

    def supervisor_node(state: AgentState) -> AgentState:
        """Node Supervisor: phân tích tin nhắn và quyết định route (hỗ trợ đa ý định)."""
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            *state["messages"]
        ]

        try:
            result = structured_llm.invoke(messages)
            agents = [a for a in result.agents if a in AGENT_MEMBERS]
            direct_response = result.direct_response or ""
        except Exception as e:
            print(f"  ⚠️ [Supervisor] Structured output failed: {e}, using fallback...")
            response = llm.invoke(messages)
            agents, direct_response = _fallback_parse(response.content)

        # ── FIX: Nếu có agents thì PHẢI xóa direct_response ──
        # LLM đôi khi trả về cả agents VÀ direct_response cùng lúc.
        # Agents phải được ưu tiên xử lý, direct_response chỉ dùng cho chào hỏi.
        if agents:
            direct_response = ""

        # Nếu không có agents và không có direct_response → mặc định chào
        if not agents and not direct_response:
            direct_response = "Dạ em chào anh/chị! Em có thể giúp gì cho mình về sản phẩm, đơn hàng, hoặc chính sách ạ? 😊"

        print(f"  🧠 [Supervisor] agents={agents}, direct={'Yes' if direct_response else 'No'}")
        return {
            "pending_agents": agents,
            "current_agent_index": 0,
            "direct_response": direct_response,
        }

    return supervisor_node


def _fallback_parse(content: str) -> tuple:
    """Fallback parsing khi structured output thất bại."""
    content = content.strip()

    # Thử parse JSON trước
    try:
        data = json.loads(content)
        agents = [a for a in data.get("agents", []) if a in AGENT_MEMBERS]
        direct = data.get("direct_response", "")
        return agents, direct
    except (json.JSONDecodeError, TypeError):
        pass

    # Regex fallback (tương tự logic cũ)
    content_lower = content.lower()
    agents = []
    if "ops" in content_lower:
        agents.append("ops_agent")
    if "sale" in content_lower:
        agents.append("sales_agent")
    if "support" in content_lower:
        agents.append("support_agent")

    return agents, ""
