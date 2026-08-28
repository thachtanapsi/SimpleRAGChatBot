"""Lỗi miền nghiệp vụ để CLI và HTTP dùng chung."""


class RAGError(Exception):
    status_code = 500


class ValidationError(RAGError):
    status_code = 400


class DuplicateDocumentError(RAGError):
    status_code = 409

    def __init__(self, document_id: str):
        super().__init__("Tài liệu này đã tồn tại")
        self.document_id = document_id


class DocumentNotFoundError(RAGError):
    status_code = 404


class DocumentBusyError(RAGError):
    status_code = 409


class ConflictError(RAGError):
    status_code = 409


class PromotionConflictError(ConflictError):
    """Stable per-document conflict detected by the SQLite promotion CAS."""

    def __init__(self, error_code: str):
        super().__init__("Full Analysis promotion conflict")
        self.error_code = error_code


class ServiceUnavailableError(RAGError):
    status_code = 503


class GenerationTruncatedError(ServiceUnavailableError):
    """Ollama stopped because the configured generation limit was reached."""

    error_code = "response_truncated"

    def __init__(self, token_limit: int):
        super().__init__(
            f"Phản hồi bị cắt vì đã chạm giới hạn sinh {token_limit} token; "
            "kết quả chưa được lưu. Vui lòng thử lại."
        )
        self.token_limit = token_limit
