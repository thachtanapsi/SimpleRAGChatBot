"""Điểm vào CLI cũ.

Chạy ``python chatbot.py`` để chat trong terminal. Các lệnh mới:

- ``python chatbot.py ingest papers``: import/index PDF rồi thoát.
- ``python chatbot.py serve``: chạy Web UI và REST API tại localhost:8000.

Mọi logic được dùng chung với web nằm trong package :mod:`rag_app`.
"""

import os
from pathlib import Path

# Khi gọi file bằng đường dẫn tuyệt đối từ thư mục khác, dữ liệu/papers vẫn nằm
# cạnh chatbot.py. Console command ``local-rag`` thì dùng thư mục terminal.
os.environ.setdefault("RAG_BASE_DIR", str(Path(__file__).resolve().parent))

from rag_app.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
