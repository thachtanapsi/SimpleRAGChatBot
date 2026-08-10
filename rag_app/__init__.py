"""Local RAG application.

Các biến offline phải được đặt trước khi Hugging Face/Transformers được import.
`setdefault` cho phép người vận hành chủ động ghi đè khi tải model lần đầu.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

__all__ = ["__version__"]
__version__ = "1.0.0"
