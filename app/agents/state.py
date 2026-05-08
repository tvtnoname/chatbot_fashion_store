"""
Định nghĩa AgentState dùng chung cho toàn bộ đồ thị LangGraph Multi-Agent.
"""
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State trung tâm được truyền giữa các Node trong đồ thị.
    - messages: Lịch sử hội thoại (tự động nối thêm nhờ add_messages).
    - next: Tên của Agent tiếp theo sẽ xử lý (do Supervisor quyết định).
    - user_id: ID người dùng hiện tại, truyền xuyên suốt đồ thị.
    - products: Danh sách sản phẩm trả về từ Sales Agent (không lưu vào messages).
    """
    messages: Annotated[list, add_messages]
    next: str
    user_id: str
    products: list
