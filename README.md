<<<<<<< HEAD
# Simple RAG Chatbot chạy hoàn toàn local

Ứng dụng hỏi đáp nội dung PDF bằng Retrieval-Augmented Generation (RAG). Toàn
bộ quá trình đọc PDF, tạo embedding, tìm kiếm vector và sinh câu trả lời đều
chạy trên máy cá nhân; chương trình không cần OpenAI API key.

## Kiến trúc

```text
PDF trong papers/
    -> PyPDFLoader
    -> RecursiveCharacterTextSplitter
    -> BAAI/bge-m3 (embedding local)
    -> FAISS (vector index trong RAM)
    -> lấy 5 chunk liên quan nhất
    -> Gemma 4 E2B qua Ollama local
    -> câu trả lời
```

Mỗi PDF ban đầu được nối thành một `Document`, sau đó mới được chia thành các
chunk 1.200 ký tự với 200 ký tự chồng lấn.

## Cấu hình đã kiểm tra

Hướng dẫn này đã chạy thành công với:

- macOS trên Apple Silicon, RAM 16 GB.
- Python 3.13.
- Ollama 0.32.5.
- `gemma4:e2b` (khoảng 7,2 GB trên ổ đĩa).
- `BAAI/bge-m3` làm embedding local.

Gemma 4 E2B là biến thể dành cho thiết bị edge. Thông tin và các tag khác có
tại [thư viện Gemma 4 của Ollama](https://ollama.com/library/gemma4).

## 1. Cài công cụ hệ thống

### Homebrew

Nếu máy chưa có Homebrew, cài theo hướng dẫn chính thức tại
[brew.sh](https://brew.sh/), sau đó kiểm tra:

```bash
brew --version
```

### Python

Cài Python 3.13 nếu máy chưa có phiên bản phù hợp:

```bash
brew install python@3.13
python3 --version
```

### Ollama

Cài và khởi động Ollama dưới dạng dịch vụ nền:

```bash
brew install ollama
brew services start ollama
ollama --version
```

Kiểm tra trạng thái dịch vụ:

```bash
brew services list | grep ollama
```

Trạng thái mong đợi là `started`.

## 2. Mở thư mục project

```bash
cd /Users/thachtan/Documents/source/APG/SimpleRAGChatBot
```

Nếu project nằm ở vị trí khác, thay đường dẫn trên bằng đường dẫn thực tế.

## 3. Tạo môi trường Python riêng

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Sau khi kích hoạt, terminal thường hiển thị `(.venv)` ở đầu dòng.

## 4. Cài Python dependencies

Cài đúng các phiên bản đã được kiểm tra với project:

```bash
python -m pip install \
  "langchain-community==0.4.2" \
  "langchain-core==1.5.3" \
  "langchain-text-splitters==1.1.2" \
  "langchain-huggingface==1.2.2" \
  "langchain-ollama==1.1.0" \
  "faiss-cpu==1.15.0" \
  "pypdf==6.15.0" \
  "sentence-transformers==5.7.0"
```

Kiểm tra các import chính:

```bash
python -c "import faiss, pypdf; from langchain_ollama import ChatOllama; print('Dependencies OK')"
```

Project không còn sử dụng `langchain-openai` hoặc `unstructured`.

## 5. Tải Gemma 4 về Ollama

Tải đúng model mà `chatbot.py` đang cấu hình:

```bash
ollama pull gemma4:e2b
```

Model có dung lượng khoảng 7,2 GB nên thời gian tải phụ thuộc tốc độ mạng. Kiểm
tra sau khi tải:

```bash
ollama list
ollama run gemma4:e2b "Trả lời đúng một từ: OK"
```

Kết quả mong đợi của lệnh thứ hai là `OK`. Model được lưu trong
`~/.ollama/models` và không cần tải lại ở những lần chạy sau.

## 6. Tải model embedding BGE-M3 lần đầu

`chatbot.py` bắt buộc Hugging Face chạy offline để tránh kiểm tra mạng trong
lúc sử dụng. Vì vậy cần tải BGE-M3 vào cache trước lần chạy đầu tiên:

```bash
python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-m3"); print("BGE-M3 downloaded")'
```

Model có dung lượng vài GB. Trong lần tải này, cảnh báo
`sending unauthenticated requests to the HF Hub` chỉ có nghĩa là chương trình
đang tải model public mà không đăng nhập; nó không tải PDF của bạn lên Hugging
Face. Sau khi hoàn tất, model nằm trong cache `~/.cache/huggingface/`.

## 7. Thêm tài liệu PDF

Tạo thư mục nếu chưa có:

```bash
mkdir -p papers
```

Chép các file cần hỏi đáp vào đó:

```text
SimpleRAGChatBot/
├── chatbot.py
├── readme.md
└── papers/
    ├── tai_lieu_1.pdf
    └── tai_lieu_2.pdf
```

Chương trình tìm tất cả file có đuôi `.pdf` bên trong `papers` và các thư mục
con của nó.

## 8. Chạy chatbot

Bảo đảm Ollama đang chạy:

```bash
brew services start ollama
```

Sau đó chạy:

```bash
source .venv/bin/activate
python chatbot.py
```

Ví dụ:

```text
Ask a question: Enterprise RAG cần đánh giá những gì?

Answer:
...
```

Nhấn `Ctrl+C` để dừng nếu chương trình đang xử lý hoặc cần thoát sớm.

## 9. Chạy từ thư mục khác

`chatbot.py` tính đường dẫn `papers` dựa trên vị trí của chính file nên có thể
chạy bằng đường dẫn đầy đủ:

```bash
/Users/thachtan/Documents/source/APG/SimpleRAGChatBot/.venv/bin/python \
  /Users/thachtan/Documents/source/APG/SimpleRAGChatBot/chatbot.py
```

## Cấu hình quan trọng

| Cấu hình | Giá trị | Ý nghĩa |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Model embedding đa ngôn ngữ chạy local |
| `CHAT_MODEL` | `gemma4:e2b` | Model tạo câu trả lời qua Ollama local |
| `chunk_size` | `1200` | Số ký tự tối đa trong một chunk |
| `chunk_overlap` | `200` | Phần nội dung lặp giữa hai chunk |
| `k` | `5` | Số chunk được đưa vào prompt |
| `num_ctx` | `8192` | Context window dùng khi chạy Gemma 4 |
| `num_predict` | `2000` | Số token sinh tối đa |

FAISS index hiện chỉ được giữ trong RAM và được tạo lại mỗi lần chạy. Khi số
lượng PDF lớn, nên bổ sung bước `save_local()`/`load_local()` để tránh tạo lại
embedding ở mỗi lần khởi động.

## Quyền riêng tư và kết nối mạng

Sau khi hai model đã được tải thành công:

- PDF và câu hỏi không được gửi tới OpenAI.
- BGE-M3 tạo embedding trên máy và bị khóa ở chế độ offline.
- Gemma 4 sinh câu trả lời qua Ollama tại `localhost:11434`.
- FAISS lưu vector trong RAM của tiến trình.
- File `.env` và `OPENAI_API_KEY` không được `chatbot.py` sử dụng.

Bạn có thể ngắt Internet sau khi tải model để xác nhận chatbot vẫn hoạt động.

## Xử lý lỗi thường gặp

### `ModuleNotFoundError`

Môi trường ảo chưa được kích hoạt hoặc dependencies chưa được cài:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
```

Sau đó chạy lại lệnh cài dependencies ở bước 4.

### `model 'gemma4:e2b' not found`

```bash
ollama pull gemma4:e2b
ollama list
```

### Không kết nối được `localhost:11434`

Khởi động lại Ollama:

```bash
brew services restart ollama
```

Kiểm tra:

```bash
ollama ps
```

### Không tìm thấy BGE-M3 trong cache offline

Kết nối Internet và chạy lại bước tải embedding:

```bash
python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-m3")'
```

### Chương trình trả về ít hoặc không đúng tài liệu

- Kiểm tra PDF có chứa text thật hay chỉ là ảnh scan.
- Thử tăng `k` từ `5` lên `8`.
- Điều chỉnh `chunk_size` và `chunk_overlap`.
- PDF scan cần thêm một bước OCR; `PyPDFLoader` không tự OCR ảnh.

### Máy dùng nhiều RAM hoặc phản hồi chậm

- Đóng các ứng dụng đang dùng nhiều bộ nhớ.
- Giảm `num_ctx` từ `8192` xuống `4096`.
- Giảm `k` hoặc `num_predict`.
- Lượt chạy đầu thường chậm hơn vì phải nạp model vào RAM/GPU.

### `DeprecationWarning` từ `langchain-community`

Đây là cảnh báo về lộ trình package, không phải lỗi thực thi. Phiên bản trong
hướng dẫn này vẫn đã được kiểm tra với project hiện tại.

## Dừng Ollama

Nếu không muốn Ollama tiếp tục chạy nền:

```bash
brew services stop ollama
```

Khởi động lại khi cần:

```bash
brew services start ollama
```

## Tài liệu tham khảo

- [Ollama trên macOS](https://docs.ollama.com/macos)
- [Gemma 4 trên Ollama](https://ollama.com/library/gemma4)
- [ChatOllama trong LangChain](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
=======
# SimpleRAGChatBot
>>>>>>> 8612f2da06b657ad16e5840f31e54b0c6a05edcc
