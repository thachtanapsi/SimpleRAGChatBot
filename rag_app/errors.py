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


class ServiceUnavailableError(RAGError):
    status_code = 503

