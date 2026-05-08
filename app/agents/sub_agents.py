"""
Định nghĩa 3 Sub-Agents chuyên biệt, mỗi Agent có prompt và tools riêng.
"""
import os
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from app.tools.api_tools import check_inventory, check_order_status


# ──────────────────────────────────────────────
# SHARED LLM (llama3.1 cho tất cả Sub-Agents)
# ──────────────────────────────────────────────
_llm = ChatOllama(model="llama3.1", temperature=0)


# ──────────────────────────────────────────────
# SUPPORT AGENT - Chuyên chính sách & quy định
# ──────────────────────────────────────────────
def create_support_agent():
    """Tạo Support Agent với RAG tool (policy_retriever)."""
    print("  → [Support Agent] Loading ChromaDB RAG...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    chroma_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")
    vector_store = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    @tool
    def policy_retriever(query: str) -> str:
        """Tìm kiếm và tra cứu các quy định, chính sách đổi trả, thanh toán của cửa hàng. Bắt buộc phải dùng tool này khi khách hỏi về quy định, chính sách."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    prompt = """Bạn là nhân viên Hỗ trợ Khách hàng (Support Agent) của cửa hàng thời trang. Luôn trả lời bằng tiếng Việt.

## NHIỆM VỤ DUY NHẤT:
Bạn CHỈ xử lý các câu hỏi về chính sách, quy định của cửa hàng (đổi trả, thanh toán, bảo hành, khuyến mãi, phí ship).

## QUY TẮC:
- BẮT BUỘC gọi tool `policy_retriever` để tra cứu chính sách.
- KHÔNG được tự bịa chính sách.
- Sau khi nhận kết quả từ tool, BẠN PHẢI đọc hiểu nội dung đó rồi DIỄN ĐẠT LẠI bằng ngôn ngữ tự nhiên, thân thiện cho khách hàng.
- CẤM trả về nguyên văn dữ liệu thô (JSON, markdown, ký tự đặc biệt như ✅ ⚖). Hãy viết lại thành câu văn hoàn chỉnh.
- TUYỆT ĐỐI KHÔNG nói "Tôi đã gọi tool..." hay nhắc đến bất kỳ công cụ nội bộ nào với khách.

## VÍ DỤ CÁCH TRẢ LỜI ĐÚNG:
Khách hỏi: "Mua về không vừa có đổi được không?"
Trả lời đúng: "Dạ được ạ! Shop hỗ trợ đổi size hoặc đổi sang mẫu khác có giá trị tương đương hoặc cao hơn. Tuy nhiên, shop không hỗ trợ hoàn tiền trong trường hợp đổi ý nhé anh/chị."
"""

    return create_react_agent(_llm, tools=[policy_retriever], prompt=prompt)


# ──────────────────────────────────────────────
# OPERATIONS AGENT - Chuyên đơn hàng
# ──────────────────────────────────────────────
def create_ops_agent():
    """Tạo Operations Agent với tools tra cứu & hủy đơn hàng."""
    print("  → [Operations Agent] Initializing...")

    prompt = """Bạn là nhân viên Quản lý Đơn hàng (Operations Agent) của cửa hàng thời trang. Luôn trả lời bằng tiếng Việt.

## NHIỆM VỤ DUY NHẤT:
Bạn CHỈ xử lý các yêu cầu liên quan đến đơn hàng: tra cứu trạng thái đơn hàng.

## QUY TẮC:
- User ID của khách luôn nằm ở đầu tin nhắn dạng "[User ID: X]". Hãy dùng số X đó làm user_id khi gọi tool.
- Nếu User ID là "Chưa đăng nhập": nhắc khách đăng nhập trước.
- Khi khách hỏi về đơn hàng: BẮT BUỘC gọi tool `check_order_status`.
- KHÔNG được tự bịa thông tin đơn hàng.
- TUYỆT ĐỐI KHÔNG nói "Tôi đã gọi tool..." hay nhắc đến tool với khách.
- CẤM trả về JSON. CẤM viết dạng {"name": ...}. Luôn trả lời bằng câu văn tiếng Việt tự nhiên.
- SAU KHI GỌI TOOL THÀNH CÔNG, BẠN PHẢI ĐỌC KẾT QUẢ VÀ TRẢ LỜI BẰNG CÂU VĂN HOÀN CHỈNH.
- Trả lời tự nhiên (Ví dụ: "Dạ, đơn hàng gần nhất của anh/chị là đơn #105, hiện đang ở trạng thái PENDING...")."""

    return create_react_agent(_llm, tools=[check_order_status], prompt=prompt)


# ──────────────────────────────────────────────
# SALES AGENT - Chuyên tồn kho & tư vấn sản phẩm
# ──────────────────────────────────────────────
def create_sales_agent():
    """Tạo Sales Agent với tool check_inventory."""
    print("  → [Sales Agent] Initializing...")

    prompt = """Bạn là nhân viên Tư vấn Bán hàng (Sales Agent) của cửa hàng thời trang. Luôn trả lời bằng tiếng Việt.

## NHIỆM VỤ DUY NHẤT:
Bạn CHỈ xử lý các câu hỏi liên quan đến sản phẩm và tồn kho: còn hàng không, size nào, màu gì, giá bao nhiêu.

## QUY TẮC:
- BẮT BUỘC gọi tool `check_inventory` để tra cứu thông tin sản phẩm.
- KHÔNG được tự bịa ra thông tin sản phẩm.
- Trả lời thân thiện, khéo léo như một nhân viên sale chuyên nghiệp.
- Nếu có hàng: khuyến khích khách mua, gợi ý thêm.
- Nếu hết hàng: xin lỗi nhẹ nhàng và gợi ý sản phẩm tương tự nếu có.
- TUYỆT ĐỐI KHÔNG nói "Tôi đã gọi tool..." hay nhắc đến tool với khách.
- CẤM trả về JSON. CẤM viết dạng {"name": ...}. Luôn trả lời bằng câu văn tiếng Việt tự nhiên."""

    return create_react_agent(_llm, tools=[check_inventory], prompt=prompt)
