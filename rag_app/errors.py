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


class AgentTimeoutError(ServiceUnavailableError):
    error_code = "agent_timeout"

    def __init__(self):
        super().__init__(
            "Luồng RAG nâng cao đã hết thời gian xử lý; kết quả chưa được lưu."
        )


class RetrievalToolsUnavailableError(ServiceUnavailableError):
    error_code = "retrieval_tools_unavailable"

    def __init__(self):
        super().__init__(
            "Các công cụ truy xuất cục bộ tạm thời không khả dụng; kết quả chưa được lưu."
        )


class SelfCheckUnavailableError(ServiceUnavailableError):
    error_code = "self_check_unavailable"

    _REASON_MESSAGES = {
        "self_check_response_truncated": (
            "Phản hồi của model kiểm chứng bị cắt trước khi hoàn tất; "
            "kết quả chưa được lưu."
        ),
        "self_check_response_parse_failed": (
            "Không đọc được JSON do model kiểm chứng trả về; "
            "kết quả chưa được lưu."
        ),
        "self_check_response_contract_invalid": (
            "Kết quả của model kiểm chứng không đúng cấu trúc bắt buộc; "
            "kết quả chưa được lưu."
        ),
        "self_check_source_scope_invalid": (
            "Model kiểm chứng tham chiếu nguồn ngoài tập bằng chứng; "
            "kết quả chưa được lưu."
        ),
        "self_check_timeout": (
            "Bước kiểm chứng đã hết thời gian xử lý; kết quả chưa được lưu."
        ),
        "self_check_execution_failed": (
            "Bước kiểm chứng gặp lỗi kỹ thuật; kết quả chưa được lưu."
        ),
    }

    def __init__(self, reason_code: str | None = None):
        if reason_code is not None and reason_code not in self._REASON_MESSAGES:
            raise ValueError("unsupported self-check reason code")
        self.reason_code = reason_code
        super().__init__(
            self._REASON_MESSAGES.get(
                reason_code,
                "Model kiểm chứng cục bộ tạm thời không khả dụng; "
                "kết quả chưa được lưu.",
            )
        )


class ReviewModelUnavailableError(ServiceUnavailableError):
    error_code = "rag_review_model_unavailable"

    def __init__(self):
        super().__init__(
            "Model planner/grader cục bộ tạm thời không khả dụng; kết quả chưa được lưu."
        )


class GenerationTruncatedError(ServiceUnavailableError):
    """Ollama stopped because the configured generation limit was reached."""

    error_code = "response_truncated"

    def __init__(self, token_limit: int):
        super().__init__(
            f"Phản hồi bị cắt vì đã chạm giới hạn sinh {token_limit} token; "
            "kết quả chưa được lưu. Vui lòng thử lại."
        )
        self.token_limit = token_limit
