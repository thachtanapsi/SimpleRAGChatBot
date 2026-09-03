# Advanced RAG release gate v1

Release gate dùng hai file JSONL local, nối với nhau bằng `id`:

- golden JSONL là ground truth được người review; không chứa output của model;
- result JSONL là trace đã gắn nhãn của đúng một lần chạy local. Trace chỉ cần ID,
  nhãn support và latency, không cần chain-of-thought hay toàn bộ answer/evidence.

Executable schema nằm trong `rag_app.evaluation.AdvancedEvalCase` và
`AdvancedEvalResult`. Cả hai record bắt buộc có `"schema_version": 1`, từ chối
field thừa và được validate strict trước khi chấm. Không có bộ 150 câu giả trong
repository; nhóm vận hành phải author và review corpus thực tế trước rollout.

## Golden JSONL

Mỗi dòng có các field sau:

```json
{
  "schema_version": 1,
  "id": "relationship-jpy-001",
  "question": "Những mã nào cùng chịu rủi ro JPY?",
  "intent": "relationship",
  "expected_route": "graph",
  "expected_tools": ["graph"],
  "answerable": true,
  "scope": {
    "document_ids": ["doc_aaa", "doc_bbb"],
    "filters": {"date_to": "2026-08-28"}
  },
  "evidence": [
    {
      "id": "ev-aaa-jpy",
      "document_id": "doc_aaa",
      "parent_id": "parent_aaa_risk",
      "quote": "<exact quote copied from the raw parent>",
      "start_offset": 120,
      "end_offset": 166,
      "quote_sha256": "<lowercase SHA-256 of the UTF-8 quote>",
      "page": 1
    }
  ],
  "claims": [
    {"id": "claim-aaa", "evidence_ids": ["ev-aaa-jpy"]}
  ],
  "entities": [
    {"id": "security:AAA", "canonical_key": "AAA"},
    {"id": "risk:jpy", "canonical_key": "jpy"}
  ],
  "relations": [
    {
      "id": "rel-aaa-jpy",
      "subject_entity_id": "security:AAA",
      "predicate": "EXPOSED_TO",
      "object_entity_id": "risk:jpy",
      "evidence_ids": ["ev-aaa-jpy"]
    }
  ],
  "simple_non_graph": false
}
```

Đây chỉ là minh họa schema, không phải một golden case. `quote` phải khớp nguyên
văn raw parent, offset là vị trí Unicode character `[start_offset, end_offset)`,
và SHA-256 phải khớp quote. Mỗi answerable case phải có ít nhất một evidence span
và một claim. Claim số liệu thêm object `numeric` với đúng ba string
`value/unit/period`. Case unanswerable không khai báo claim/relation được hỗ trợ.

Corpus release phải có ít nhất 150 ID duy nhất và có coverage cho answerable,
unanswerable, numeric, graph entity, graph relation và simple non-graph. Mỗi
evidence document phải nằm trong caller scope; mọi claim/relation chỉ được tham
chiếu evidence ID đã khai báo. Predicate graph bị giới hạn bằng ontology v1.

## Result JSONL

Mỗi dòng kết quả có cùng `id` và các field:

- `route`, `tools`;
- `candidate_evidence_ids` theo thứ tự retrieval và `context_evidence_ids` theo
  thứ tự context. Evaluator chỉ lấy lần lượt top 20 và top 5;
- `claims[]`: `claim_id`, nhãn `support`, `citation_evidence_ids`, và `numeric`
  nếu có;
- `citations[]`: citation ID, document ID, evidence ID nếu map được và nhãn
  `supported` do rule/human audit tạo;
- `retrieved_document_ids` gồm document đã được bất kỳ tool nào trả về;
  `scope_filter_violations` là số candidate vi phạm caller date/source/section
  filter. Hai field này đo scope escape trên retrieval, không chỉ citation;
- `abstained`, `latency_ms`, `timed_out`, `error_code`;
- `resolved_entity_ids`, `graph_relations`, `graph_evidence_ids` (evaluator lấy
  top 10);
- `baseline_latency_ms` cho mọi case có `simple_non_graph=true`. Baseline phải
  đo cùng case/corpus khi advanced flags tắt.

Result trace phải được tạo sau khi answer đã verify/buffer. Không ghi query,
answer, raw evidence hay chain-of-thought vào trace. Mapping source sang evidence
ID dùng exact document/parent/span, không dùng graph ID hoặc inferred path.

## Lệnh

Chỉ kiểm tra schema, link ground truth, size và coverage; không khởi động runtime:

```bash
python chatbot.py evaluate /path/advanced_questions.jsonl \
  --release-gate --validate-only
```

Chấm một labelled run và regression 30 câu:

```bash
python chatbot.py evaluate /path/advanced_questions.jsonl \
  --release-gate \
  --results /path/advanced_results.jsonl \
  --legacy-results /path/current_30_result.json \
  --legacy-baseline evals/hybrid_full.json
```

Chạy lệnh eval 30 câu hiện tại và lưu JSON output để tạo `current_30_result.json`.
Evaluator thêm canonical `question_set_sha256` (không phụ thuộc whitespace/line
ending); current và baseline phải có cùng đúng 30 cases và cùng hash. Baseline
được commit tại `evals/hybrid_full.json`. No-regression
so sánh `parent_recall_at_5`, `citation_correct_page`, `abstention_accuracy` và
`answers_with_citation` theo chiều không giảm; `irrelevant_citations` theo chiều
không tăng. Current run cũng không được xuất hiện failed case ID mới ngoài danh
sách đã có trong baseline. Đây là so sánh tương đối với lần chạy đã khóa, không
phải yêu cầu baseline lịch sử đạt acceptance tuyệt đối.

Exit code `0` chỉ khi tất cả gate đạt, `2` khi schema hợp lệ nhưng quality gate
không đạt. Schema/case-result mismatch là lỗi validation rõ ràng. Chế độ này
không gọi model judge, web hay cloud.

Các ngưỡng được version trong `ADVANCED_RELEASE_THRESHOLDS`: Candidate Recall@20
0.95, final context Recall@5 0.90, citation/claim support 0.95, numeric exactness
1.0, abstention 0.90, route 0.95, graph entity 0.95, relation precision 0.90,
graph evidence Recall@10 0.85; unsupported citation và scope escape phải bằng 0.
Simple non-graph P95 không vượt 2 lần baseline, mọi request phải dừng trước 120
giây, không có runtime error, và kết quả 30 câu không được giảm chất lượng so với
baseline đã commit.
