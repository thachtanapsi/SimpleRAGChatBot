"""Smoke test thủ công với model/index thật; không được pytest tự động gọi."""

from __future__ import annotations

import asyncio

from rag_app.runtime import Runtime


async def main() -> None:
    runtime = Runtime()
    await runtime.start()
    try:
        for _ in range(600):
            documents = runtime.storage.list_documents()
            if documents and not any(
                item["status"] in {"pending", "indexing"} for item in documents
            ):
                break
            await asyncio.sleep(0.25)
        response = await runtime.chat.answer(
            "What metrics should be used to evaluate enterprise RAG?"
        )
        print("ollama:", runtime.ollama_ready)
        print("documents:", [(item["filename"], item["status"]) for item in documents])
        print("answer:", response["answer"])
        print(
            "citations:",
            [
                (item["filename"], item["page"], item["cited"])
                for item in response["citations"]
            ],
        )
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())

