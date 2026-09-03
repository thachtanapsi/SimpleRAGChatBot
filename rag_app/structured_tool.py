"""Canonical Full Analysis tool with deterministic decision rendering."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping, Sequence

from .full_reports import DECISION_STRUCTURED_SECTION, FULL_REPORT_V1
from .orchestration import (
    DeterministicAnswerRequest,
    DeterministicDecision,
    SoftFilterConflict,
    ToolName,
    ToolRequest,
)
from .retrieval import RetrievedSource
from .self_check import GeneratedAnswer, GeneratedClaim
from .storage import SQLiteStorage


class StructuredDecisionTool:
    """Read only canonical ``decision_json`` and its raw indexed parent."""

    name: ToolName = "structured"

    def __init__(self, storage: SQLiteStorage):
        self.storage = storage

    def _retrieve_sync(self, request: ToolRequest) -> list[RetrievedSource]:
        try:
            filters = request.effective_retrieval_filters()
        except SoftFilterConflict:
            return []
        hard_sections = set(request.scope.filters.digest_sections)
        if hard_sections and DECISION_STRUCTURED_SECTION not in hard_sections:
            return []
        soft_section = request.soft_filters.section if request.round == 1 else None
        if soft_section and soft_section != DECISION_STRUCTURED_SECTION:
            return []
        document_ids = list(request.scope.allowed_document_ids)
        if not document_ids:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        clauses = [
            f"d.id IN ({placeholders})",
            "d.status='ready'",
            "d.source_type='trading_digest'",
            "d.content_kind=?",
            "p.digest_section=?",
            "p.section_status IN ('complete', 'partial')",
        ]
        params: list[Any] = [
            *document_ids,
            FULL_REPORT_V1,
            DECISION_STRUCTURED_SECTION,
        ]
        configured = filters or {}
        if configured.get("tickers"):
            values = list(configured["tickers"])
            ticker_placeholders = ",".join("?" for _ in values)
            clauses.append(f"d.ticker IN ({ticker_placeholders})")
            params.extend(values)
        if configured.get("date_from"):
            clauses.append("d.analysis_date>=?")
            params.append(configured["date_from"])
        if configured.get("date_to"):
            clauses.append("d.analysis_date<=?")
            params.append(configured["date_to"])
        if request.round == 1 and request.soft_filters.latest_only:
            clauses.append(
                """
                NOT EXISTS (
                    SELECT 1 FROM documents newer
                    WHERE newer.status='ready'
                      AND newer.source_type='trading_digest'
                      AND newer.content_kind='full_report_v1'
                      AND newer.ticker=d.ticker
                      AND COALESCE(newer.analysis_cutoff, newer.analysis_date)
                          > COALESCE(d.analysis_cutoff, d.analysis_date)
                      AND newer.id IN (
                """
                + placeholders
                + "))"
            )
            params.extend(document_ids)
        with self.storage.connect() as db:
            rows = db.execute(
                f"""
                SELECT p.*, d.filename, d.source_type, d.ticker,
                       d.analysis_date, d.analysis_cutoff, d.liquidity_rank,
                       d.research_mode, d.data_provenance, d.content_kind,
                       d.report_schema, d.full_report_hash, d.pipeline_version,
                       d.execution_generation, d.contains_decision_content,
                       frp.decision_status
                FROM parents p
                JOIN documents d ON d.id=p.document_id
                JOIN full_report_payloads frp ON frp.document_id=d.id
                WHERE {' AND '.join(clauses)}
                ORDER BY COALESCE(d.analysis_cutoff, d.analysis_date) DESC,
                         d.ticker, p.start_index, p.id
                LIMIT ?
                """,
                (*params, min(request.limit, 20)),
            ).fetchall()
        return [
            RetrievedSource(
                id=str(index),
                document_id=str(row["document_id"]),
                filename=str(row["filename"]),
                page=int(row["page"]),
                page_end=int(row["page_end"] or row["page"]),
                parent_id=str(row["id"]),
                text=str(row["text"]),
                score=0.0,
                kind=str(row["kind"] or "page"),
                source_type="trading_digest",
                ticker=row["ticker"],
                analysis_date=row["analysis_date"],
                analysis_cutoff=row["analysis_cutoff"],
                digest_section=row["digest_section"],
                liquidity_rank=row["liquidity_rank"],
                research_mode=str(row["research_mode"]),
                data_provenance=str(row["data_provenance"]),
                content_kind=str(row["content_kind"]),
                report_schema=str(row["report_schema"]),
                report_stage=row["report_stage"],
                section_status=str(row["decision_status"]),
                full_report_hash=row["full_report_hash"],
                pipeline_version=row["pipeline_version"],
                execution_generation=row["execution_generation"],
                contains_decision_content=bool(row["contains_decision_content"]),
            )
            for index, row in enumerate(rows, start=1)
        ]

    async def retrieve(self, request: ToolRequest) -> Sequence[RetrievedSource]:
        return await asyncio.to_thread(self._retrieve_sync, request)

    @staticmethod
    def _number(value: Any) -> str:
        return str(value)

    @staticmethod
    def _claim(
        text: str,
        citation_ids: Sequence[str],
        *,
        ticker: str | None = None,
        numeric: Any | None = None,
        unit: str | None = None,
        period: str | None = None,
        date: str | None = None,
    ) -> GeneratedClaim:
        return GeneratedClaim(
            text=text,
            citation_ids=list(citation_ids),
            ticker=ticker,
            numeric=(str(numeric) if numeric is not None else None),
            unit=unit,
            period=period,
            date=date,
        )

    @staticmethod
    def _citation_label(citation_ids: Sequence[str]) -> str:
        return "[" + ", ".join(citation_ids) + "]"

    @classmethod
    def _field_citations(
        cls,
        evidence: Sequence[Any],
        field_name: str,
        value: Any,
    ) -> list[str]:
        fragment = cls._decision_fragment(field_name, value)
        return list(
            dict.fromkeys(item.id for item in evidence if fragment in item.text)
        )

    @staticmethod
    def _decision_fragment(field_name: str, value: Any) -> str:
        return (
            json.dumps(field_name, ensure_ascii=False)
            + ":"
            + json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _render_decision(
        self,
        *,
        evidence: Sequence[Any],
        document: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> tuple[list[str], list[GeneratedClaim], int, bool]:
        ticker = str(document.get("ticker") or "").upper() or None
        analysis_date = str(document.get("analysis_date") or "") or None
        portfolio = decision.get("portfolio") or {}
        trading = decision.get("trading") or {}
        lines: list[str] = []
        claims: list[GeneratedClaim] = []
        rendered_fields = 0
        missing_evidence = False

        if not evidence:
            return lines, claims, rendered_fields, True
        header_citations = [evidence[0].id]
        header_label = self._citation_label(header_citations)

        header = "Quyết định"
        if ticker:
            header += f" cho {ticker}"
        if analysis_date:
            header += f" ngày {analysis_date}"
        header += f" {header_label}"
        lines.append(header)
        claims.append(
            self._claim(
                header,
                header_citations,
                ticker=ticker,
                date=analysis_date,
            )
        )

        missing = object()

        def add(
            label: str,
            field_name: str,
            value: Any,
            *,
            source_value: Any = missing,
            extra_fields: Sequence[tuple[str, Any]] = (),
            **audit: Any,
        ) -> None:
            nonlocal rendered_fields, missing_evidence
            if value is None or value == "":
                return
            lookup_value = value if source_value is missing else source_value
            citation_ids = self._field_citations(
                evidence, field_name, lookup_value
            )
            for extra_name, extra_value in extra_fields:
                if extra_value is not None:
                    citation_ids.extend(
                        self._field_citations(
                            evidence, extra_name, extra_value
                        )
                    )
            citation_ids = list(dict.fromkeys(citation_ids))
            if not citation_ids:
                missing_evidence = True
                return
            citation_label = self._citation_label(citation_ids)
            text = f"- {label}: {value} {citation_label}"
            lines.append(text)
            claims.append(
                self._claim(text, citation_ids, ticker=ticker, **audit)
            )
            rendered_fields += 1

        add("Khuyến nghị danh mục", "rating", portfolio.get("rating"))
        add("Hành động giao dịch", "action", trading.get("action"))
        add("Tóm tắt", "executive_summary", portfolio.get("executive_summary"))
        add(
            "Luận điểm đầu tư",
            "investment_thesis",
            portfolio.get("investment_thesis"),
        )
        add("Khung thời gian", "time_horizon", portfolio.get("time_horizon"))
        target = portfolio.get("price_target")
        currency = portfolio.get("price_target_currency")
        if target is not None:
            rendered = f"{self._number(target)}{f' {currency}' if currency else ''}"
            add(
                "Giá mục tiêu",
                "price_target",
                rendered,
                source_value=target,
                extra_fields=(("price_target_currency", currency),),
                numeric=target,
                unit=currency,
            )
        elif portfolio.get("price_target_unavailable_reason"):
            add(
                "Giá mục tiêu chưa có",
                "price_target_unavailable_reason",
                portfolio.get("price_target_unavailable_reason"),
            )
        add("Lý do giao dịch", "reasoning", trading.get("reasoning"))
        for label, key in (
            ("Giá vào lệnh", "entry_price"),
            ("Dừng lỗ", "stop_loss"),
        ):
            value = trading.get(key)
            if value is not None:
                add(
                    label,
                    key,
                    self._number(value),
                    source_value=value,
                    numeric=value,
                )
        add(
            "Quy mô vị thế",
            "position_sizing",
            trading.get("position_sizing"),
        )
        return lines, claims, rendered_fields, missing_evidence

    async def deterministic_answer(
        self, request: DeterministicAnswerRequest
    ) -> DeterministicDecision | None:
        allowed_document_ids = frozenset(request.scope.allowed_document_ids)
        evidence_by_document: dict[str, list[Any]] = {}
        for item in request.evidence:
            if item.document_id in allowed_document_ids:
                evidence_by_document.setdefault(item.document_id, []).append(item)
        rendered: list[str] = []
        claims: list[GeneratedClaim] = []
        statuses: list[str] = []
        for document_id, document_evidence in evidence_by_document.items():
            payload = await asyncio.to_thread(
                self.storage.get_full_report_payload, document_id
            )
            document = await asyncio.to_thread(self.storage.get_document, document_id)
            if payload is None or document is None:
                continue
            try:
                decision = json.loads(str(payload["decision_json"]))
            except (TypeError, ValueError):
                continue
            status = str(payload.get("decision_status") or decision.get("status"))
            if status not in {"complete", "partial"}:
                continue
            (
                lines,
                decision_claims,
                rendered_fields,
                missing_evidence,
            ) = self._render_decision(
                evidence=document_evidence,
                document=document,
                decision=decision,
            )
            if rendered_fields == 0:
                continue
            rendered.append("\n".join(lines))
            claims.extend(decision_claims)
            statuses.append(
                "partial" if status == "partial" or missing_evidence else "complete"
            )
        if not rendered:
            return None
        status = "partial" if "partial" in statuses else "complete"
        qualification = (
            "Lưu ý: quyết định có cấu trúc chưa hoàn chỉnh hoặc chưa đủ "
            "raw parent evidence cho mọi trường."
            if status == "partial"
            else None
        )
        return DeterministicDecision(
            generated=GeneratedAnswer(
                answer="\n\n".join(rendered),
                claims=claims,
            ),
            status=status,
            qualification=qualification,
        )


__all__ = ["StructuredDecisionTool"]
