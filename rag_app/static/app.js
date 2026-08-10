"use strict";

const state = { sessionId: null, busy: false };
const elements = {
  documents: document.querySelector("#documents"),
  uploadForm: document.querySelector("#upload-form"),
  uploadStatus: document.querySelector("#upload-status"),
  chatForm: document.querySelector("#chat-form"),
  question: document.querySelector("#question"),
  send: document.querySelector("#send"),
  messages: document.querySelector("#messages"),
  health: document.querySelector("#health"),
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
  const checked = [...document.querySelectorAll(".document-select:checked")];
  return checked.map((item) => item.value);
}

async function loadDocuments() {
  const documents = await api("/api/documents");
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

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "document-select";
    checkbox.value = item.id;
    checkbox.checked = item.status === "ready";
    checkbox.disabled = item.status !== "ready";
    checkbox.setAttribute("aria-label", `Chọn ${item.filename}`);

    const name = document.createElement("span");
    name.className = "document-name";
    name.textContent = item.filename;

    const remove = document.createElement("button");
    remove.className = "delete-document";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Xóa tài liệu";
    remove.disabled = item.status === "indexing";
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Xóa ${item.filename}?`)) return;
      try { await api(`/api/documents/${encodeURIComponent(item.id)}`, { method: "DELETE" }); await loadDocuments(); }
      catch (error) { window.alert(error.message); }
    });

    const meta = document.createElement("p");
    meta.className = "document-meta";
    const badge = document.createElement("span");
    badge.className = `badge ${item.status}`;
    badge.textContent = item.status;
    meta.append(badge, document.createTextNode(` · ${item.page_count || "?"} trang · ${formatBytes(item.size_bytes)}`));
    if (item.error) meta.append(document.createTextNode(` · ${item.error}`));
    top.append(checkbox, name, remove);
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
  bubble.textContent = text;
  const citations = document.createElement("div");
  citations.className = "citations";
  wrapper.append(label, bubble, citations);
  elements.messages.append(wrapper);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return { wrapper, bubble, citations };
}

function renderCitations(container, citations) {
  container.replaceChildren();
  for (const source of citations.filter((item) => item.cited)) {
    const chip = document.createElement("span");
    chip.className = "citation";
    chip.textContent = `[${source.id}] ${source.filename} · trang ${source.page}`;
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
  try {
    const selected = selectedDocumentIds();
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        session_id: state.sessionId,
        document_ids: selected.length ? selected : undefined,
      }),
    });
    await consumeSse(response, {
      meta(data) { state.sessionId = data.session_id; },
      token(data) { assistant.bubble.textContent += data.text; elements.messages.scrollTop = elements.messages.scrollHeight; },
      sources(data) { renderCitations(assistant.citations, data.citations || []); },
      done(data) { if (data.answer) assistant.bubble.textContent = data.answer; },
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
    await loadDocuments();
    setInterval(loadDocuments, 2500);
  } catch (error) { elements.health.textContent = `Không kết nối được: ${error.message}`; }
}

initialise();

