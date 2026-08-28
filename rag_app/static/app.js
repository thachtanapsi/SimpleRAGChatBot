"use strict";

const state = { sessionId: null, busy: false, documentPage: 1, documentTotal: 0, latestBatchId: null, selectedDocuments: new Set() };
const elements = {
  documents: document.querySelector("#documents"),
  uploadForm: document.querySelector("#upload-form"),
  uploadStatus: document.querySelector("#upload-status"),
  chatForm: document.querySelector("#chat-form"),
  question: document.querySelector("#question"),
  send: document.querySelector("#send"),
  messages: document.querySelector("#messages"),
  health: document.querySelector("#health"),
  filters: document.querySelector("#document-filters"),
  sourceType: document.querySelector("#source-type"),
  contentKindFilter: document.querySelector("#content-kind-filter"),
  tickerFilter: document.querySelector("#ticker-filter"),
  dateFrom: document.querySelector("#date-from"),
  dateTo: document.querySelector("#date-to"),
  sectionFilter: document.querySelector("#section-filter"),
  reportStageFilter: document.querySelector("#report-stage-filter"),
  researchModeFilter: document.querySelector("#research-mode-filter"),
  dataProvenanceFilter: document.querySelector("#data-provenance-filter"),
  activeBatch: document.querySelector("#active-batch"),
  pageLabel: document.querySelector("#document-page"),
  previousDocuments: document.querySelector("#previous-documents"),
  nextDocuments: document.querySelector("#next-documents"),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* response không phải JSON */ }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

function formatBytes(value) {
  if (!value) return "0 KB";
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function selectedDocumentIds() {
  return [...state.selectedDocuments];
}

async function loadDocuments() {
  const query = new URLSearchParams({ page: String(state.documentPage), page_size: "50" });
  if (elements.sourceType.value) query.set("source_type", elements.sourceType.value);
  if (elements.contentKindFilter.value) query.set("content_kind", elements.contentKindFilter.value);
  if (elements.tickerFilter.value.trim()) query.set("ticker", elements.tickerFilter.value.trim().toUpperCase());
  if (elements.dateFrom.value) query.set("date_from", elements.dateFrom.value);
  if (elements.dateTo.value) query.set("date_to", elements.dateTo.value);
  if (elements.sectionFilter.value) query.set("digest_section", elements.sectionFilter.value);
  if (elements.reportStageFilter.value) query.set("report_stage", elements.reportStageFilter.value);
  if (elements.researchModeFilter.value) query.set("research_mode", elements.researchModeFilter.value);
  if (elements.dataProvenanceFilter.value) query.set("data_provenance", elements.dataProvenanceFilter.value);
  if (usesLatestDigestBatch()) query.set("batch_id", state.latestBatchId);
  const result = await api(`/api/documents?${query}`);
  const documents = result.items;
  state.documentTotal = result.total;
  const pageCount = Math.max(1, Math.ceil(result.total / result.page_size));
  elements.pageLabel.textContent = `Trang ${result.page}/${pageCount} · ${result.total}`;
  elements.previousDocuments.disabled = result.page <= 1;
  elements.nextDocuments.disabled = result.page >= pageCount;
  elements.documents.replaceChildren();
  if (!documents.length) {
    const empty = document.createElement("p");
    empty.className = "status";
    empty.textContent = "Chưa có tài liệu.";
    elements.documents.append(empty);
    return;
  }
  for (const item of documents) {
    const card = document.createElement("div");
    card.className = "document";
    const top = document.createElement("div");
    top.className = "document-top";

    const selector = document.createElement("span");
    if (item.source_type === "pdf") {
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "document-select";
      checkbox.value = item.id;
      checkbox.checked = state.selectedDocuments.has(item.id);
      checkbox.disabled = item.status !== "ready";
      checkbox.setAttribute("aria-label", `Chọn ${item.filename}`);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) state.selectedDocuments.add(item.id);
        else state.selectedDocuments.delete(item.id);
      });
      selector.append(checkbox);
    } else {
      selector.className = "digest-marker";
      selector.classList.toggle("full", item.content_kind === "full_report_v1");
      selector.textContent = item.content_kind === "full_report_v1"
        ? "FULL"
        : (item.liquidity_rank ? `#${item.liquidity_rank}` : "D");
    }

    const name = document.createElement("span");
    name.className = "document-name";
    name.textContent = item.filename;

    const remove = document.createElement("button");
    remove.className = "delete-document";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = item.source_type === "trading_digest" ? "Research document được giữ vĩnh viễn" : "Xóa tài liệu";
    remove.disabled = item.status === "indexing" || item.source_type === "trading_digest";
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Xóa ${item.filename}?`)) return;
      try { await api(`/api/documents/${encodeURIComponent(item.id)}`, { method: "DELETE" }); state.selectedDocuments.delete(item.id); await loadDocuments(); }
      catch (error) { window.alert(error.message); }
    });

    const meta = document.createElement("p");
    meta.className = "document-meta";
    const badge = document.createElement("span");
    badge.className = `badge ${item.status}`;
    badge.textContent = item.status;
    const historicalLabel = item.data_provenance === "historical_replay" ? " · dựng lại lịch sử" : "";
    const analysisLabel = item.content_kind === "full_report_v1" ? " · Full Analysis" : " · Daily digest";
    const details = item.source_type === "trading_digest"
      ? `${analysisLabel} · ${item.analysis_date || "?"} · ${item.ticker || "?"}${item.digest_status === "partial" ? " · partial" : ""}${historicalLabel}`
      : ` · ${item.page_count || "?"} trang · ${formatBytes(item.size_bytes)}`;
    meta.append(badge, document.createTextNode(details));
    if (item.content_kind === "full_report_v1" && item.contains_decision_content) {
      const decisionBadge = document.createElement("span");
      decisionBadge.className = "decision-content-badge";
      decisionBadge.textContent = "Full analysis · có nội dung quyết định giao dịch";
      decisionBadge.title = "Nội dung phân tích; không phải lệnh đã được đặt";
      meta.append(document.createTextNode(" · "), decisionBadge);
      const disclaimer = document.createElement("span");
      disclaimer.className = "decision-disclaimer";
      disclaimer.textContent = " · Không phải lệnh đã được đặt";
      meta.append(disclaimer);
    }
    if (item.error) meta.append(document.createTextNode(` · ${item.error}`));
    top.append(selector, name, remove);
    card.append(top, meta);
    elements.documents.append(card);
  }
}

function appendMessage(role, text = "") {
  const welcome = elements.messages.querySelector(".welcome");
  if (welcome) welcome.remove();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "role";
  label.textContent = role === "user" ? "Bạn" : "Gemma";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.classList.add("markdown-body");
    renderAssistantAnswer(bubble, text);
  } else {
    bubble.textContent = text;
  }
  const citations = document.createElement("div");
  citations.className = "citations";
  wrapper.append(label, bubble, citations);
  elements.messages.append(wrapper);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return { wrapper, bubble, citations };
}

function renderAssistantAnswer(bubble, markdown) {
  if (window.RagMarkdown && typeof window.RagMarkdown.render === "function") {
    bubble.innerHTML = window.RagMarkdown.render(markdown);
  } else {
    bubble.textContent = markdown;
  }
}

function renderCitations(container, citations) {
  container.replaceChildren();
  for (const source of citations.filter((item) => item.cited)) {
    const chip = document.createElement("span");
    chip.className = "citation";
    if (source.source_type === "trading_digest") {
      if (source.content_kind === "full_report_v1") {
        chip.classList.add("full-analysis");
        const kind = document.createElement("strong");
        kind.className = "citation-kind";
        kind.textContent = "Full Analysis";
        chip.append(kind, document.createTextNode(`[${source.id}] ${source.label}`));
      } else {
        chip.textContent = `[${source.id}] ${source.label}`;
      }
    } else {
      const pageEnd = source.page_end || source.page;
      const pageLabel = pageEnd === source.page ? `${source.page}` : `${source.page}–${pageEnd}`;
      chip.textContent = `[${source.id}] ${source.filename} · trang ${pageLabel}`;
    }
    chip.title = source.snippet;
    container.append(chip);
  }
}

async function consumeSse(response, handlers) {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      let event = "message";
      let data = {};
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        if (line.startsWith("data: ")) data = JSON.parse(line.slice(6));
      }
      if (handlers[event]) handlers[event](data);
    }
    if (done) break;
  }
}

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = elements.uploadForm.querySelector("button");
  const input = elements.uploadForm.querySelector("input");
  const form = new FormData();
  form.append("file", input.files[0]);
  button.disabled = true;
  elements.uploadStatus.className = "status";
  elements.uploadStatus.textContent = "Đang upload…";
  try {
    await api("/api/documents", { method: "POST", body: form });
    elements.uploadStatus.textContent = "Đã nhận file, đang index.";
    input.value = "";
    await loadDocuments();
  } catch (error) {
    elements.uploadStatus.className = "status error";
    elements.uploadStatus.textContent = error.message;
  } finally { button.disabled = false; }
});

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const question = elements.question.value.trim();
  if (!question) return;
  state.busy = true;
  elements.send.disabled = true;
  elements.question.value = "";
  appendMessage("user", question);
  const assistant = appendMessage("assistant", "");
  let answer = "";
  try {
    const selected = selectedDocumentIds();
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        session_id: state.sessionId,
        document_ids: selected.length ? selected : undefined,
        filters: activeRetrievalFilters(),
      }),
    });
    await consumeSse(response, {
      meta(data) { state.sessionId = data.session_id; },
      token(data) {
        answer += data.text;
        renderAssistantAnswer(assistant.bubble, answer);
        elements.messages.scrollTop = elements.messages.scrollHeight;
      },
      sources(data) { renderCitations(assistant.citations, data.citations || []); },
      done(data) {
        if (typeof data.answer === "string") answer = data.answer;
        renderAssistantAnswer(assistant.bubble, answer);
      },
      error(data) { throw new Error(data.detail || "Lỗi streaming"); },
    });
  } catch (error) {
    assistant.bubble.textContent = `Lỗi: ${error.message}`;
  } finally {
    state.busy = false;
    elements.send.disabled = false;
    elements.question.focus();
  }
});

document.querySelector("#new-chat").addEventListener("click", () => {
  state.sessionId = null;
  elements.messages.replaceChildren();
  const welcome = document.createElement("div");
  welcome.className = "welcome";
  const title = document.createElement("h3");
  title.textContent = "Hội thoại mới";
  const text = document.createElement("p");
  text.textContent = "Lịch sử cũ vẫn được lưu local cho đến khi xóa qua API.";
  welcome.append(title, text);
  elements.messages.append(welcome);
});

document.querySelector("#refresh-documents").addEventListener("click", loadDocuments);
elements.filters.addEventListener("submit", async (event) => {
  event.preventDefault();
  state.documentPage = 1;
  state.selectedDocuments.clear();
  await loadDocuments();
});
elements.previousDocuments.addEventListener("click", async () => {
  if (state.documentPage > 1) state.documentPage -= 1;
  await loadDocuments();
});
elements.nextDocuments.addEventListener("click", async () => {
  if (state.documentPage * 50 < state.documentTotal) state.documentPage += 1;
  await loadDocuments();
});
elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

async function initialise() {
  try {
    const health = await api("/api/health");
    elements.health.textContent = health.status === "ok" ? "Sẵn sàng · chỉ chạy trên máy này" : "Degraded · kiểm tra Ollama/model";
    try {
      const latest = await api("/api/digest-batches/latest");
      state.latestBatchId = latest.id;
      const provenance = latest.data_provenance === "historical_replay" ? " · dựng lại lịch sử" : "";
      elements.activeBatch.textContent = `Batch mặc định: ${latest.analysis_date} · ${latest.rag_ready}/${latest.target_count} ready${provenance}`;
    } catch (_) {
      elements.activeBatch.textContent = "Chưa có daily digest; có thể chuyển sang PDF.";
    }
    await loadDocuments();
    setInterval(loadDocuments, 5000);
  } catch (error) { elements.health.textContent = `Không kết nối được: ${error.message}`; }
}

function activeRetrievalFilters() {
  const filters = {};
  if (elements.sourceType.value) filters.source_types = [elements.sourceType.value];
  if (elements.contentKindFilter.value) filters.content_kinds = [elements.contentKindFilter.value];
  if (elements.tickerFilter.value.trim()) filters.tickers = [elements.tickerFilter.value.trim().toUpperCase()];
  if (elements.dateFrom.value) filters.date_from = elements.dateFrom.value;
  if (elements.dateTo.value) filters.date_to = elements.dateTo.value;
  if (elements.sectionFilter.value) filters.digest_sections = [elements.sectionFilter.value];
  if (elements.reportStageFilter.value) filters.report_stages = [elements.reportStageFilter.value];
  if (elements.researchModeFilter.value) filters.research_modes = [elements.researchModeFilter.value];
  if (elements.dataProvenanceFilter.value) filters.data_provenances = [elements.dataProvenanceFilter.value];
  if (usesLatestDigestBatch()) filters.batch_ids = [state.latestBatchId];
  return Object.keys(filters).length ? filters : undefined;
}

function usesLatestDigestBatch() {
  return elements.sourceType.value === "trading_digest"
    && Boolean(state.latestBatchId)
    && !elements.dateFrom.value
    && !elements.dateTo.value
    && !elements.researchModeFilter.value
    && !elements.dataProvenanceFilter.value;
}

initialise();
