"""Chatbot RAG chạy hoàn toàn trên máy cá nhân.

Luồng xử lý:
1. Đọc các file PDF trong thư mục ``papers``.
2. Chia nội dung thành các đoạn nhỏ có phần chồng lấn.
3. Tạo embedding local bằng BGE-M3 và lưu vector trong FAISS (RAM).
4. Tìm 5 đoạn liên quan nhất rồi gửi chúng cho Gemma 4 qua Ollama local.

Chuẩn bị trước khi chạy:
    brew services start ollama
    ollama pull gemma4:e2b

Chạy chương trình:
    ./.venv/bin/python chatbot.py

BGE-M3 phải được tải sẵn trong Hugging Face cache. Chương trình bật chế độ
offline nên sẽ báo lỗi thay vì tự kết nối Internet nếu cache chưa có model.
"""

import os
from pathlib import Path

# Các biến này phải được đặt trước khi import Hugging Face/Transformers.
# Chúng ngăn thư viện kiểm tra phiên bản hoặc tải dữ liệu từ Hugging Face Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


EMBEDDING_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "gemma4:e2b"

# Tính đường dẫn theo vị trí chatbot.py để lệnh vẫn chạy đúng khi terminal
# đang đứng ở một thư mục khác.
papers_dir = Path(__file__).resolve().parent / "papers"

# mode="single" nối toàn bộ trang của mỗi PDF thành một Document. Sau đó
# RecursiveCharacterTextSplitter sẽ chia Document này thành các chunk cho RAG.
loader = DirectoryLoader(
    path=str(papers_dir),
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    loader_kwargs={"mode": "single"},
    show_progress=True,
    use_multithreading=True,
)
docs = loader.load()

# Thứ tự ưu tiên vị trí cắt: tiêu đề Markdown, hàng phân cách, đoạn, dòng,
# khoảng trắng, rồi cuối cùng mới cắt ở bất kỳ ký tự nào.
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

# chunk_size được tính theo số ký tự (không phải token). Phần overlap giúp câu
# nằm sát ranh giới chunk vẫn giữ được ngữ cảnh ở cả hai chunk liên tiếp.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
    is_separator_regex=True,
)

splits = text_splitter.split_documents(docs)

# BGE-M3 biến mỗi chunk thành vector ngay trên máy. Chuẩn hóa vector giúp phép
# so sánh khoảng cách trong FAISS tương ứng tốt hơn với cosine similarity.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)

# Index chỉ tồn tại trong RAM và được tạo lại mỗi lần chạy chương trình.
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# Prompt buộc model chỉ trả lời từ các chunk được retriever cung cấp. Đây là
# lớp kiểm soát hallucination; nó không thay thế việc đánh giá retrieval.
template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


prompt = ChatPromptTemplate.from_template(template)

# Gemma 4 chạy qua Ollama tại localhost:11434. num_ctx giới hạn cửa sổ ngữ
# cảnh để tiết kiệm RAM; num_predict giới hạn độ dài câu trả lời tối đa.
llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    num_ctx=8192,
    num_predict=2000,
)

# LCEL fan-out cùng một câu hỏi theo hai nhánh:
# - retriever tìm context liên quan;
# - RunnablePassthrough giữ nguyên câu hỏi cho prompt.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = input("Ask a question: ")

answer = rag_chain.invoke(question)

print("\nAnswer:\n")
print(answer)
