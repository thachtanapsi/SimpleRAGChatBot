"use strict";

(function exposeMarkdownRenderer(root, factory) {
  const renderer = factory();
  if (typeof module === "object" && module.exports) module.exports = renderer;
  if (root) root.RagMarkdown = renderer;
})(typeof globalThis === "undefined" ? null : globalThis, function createMarkdownRenderer() {
  const TOKEN_OPEN = "\u0001";
  const TOKEN_CLOSE = "\u0002";

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function safeUrl(value) {
    const url = String(value).trim();
    return /^(?:https?:|mailto:)/i.test(url) ? url : null;
  }

  function renderInline(value) {
    const tokens = [];
    const stash = (html) => {
      const token = `${TOKEN_OPEN}${tokens.length}${TOKEN_CLOSE}`;
      tokens.push(html);
      return token;
    };

    let text = String(value ?? "").replace(/[\u0001\u0002]/g, "");

    // Protect escaped Markdown characters and code before processing emphasis.
    text = text.replace(/\\([\\`*_[\]{}()#+.!|>~-])/g, (_match, character) => (
      stash(escapeHtml(character))
    ));
    text = text.replace(/(`+)([\s\S]*?)\1/g, (_match, _ticks, code) => {
      const normalized = code.replace(/^ | $/g, " ").trim();
      return stash(`<code>${escapeHtml(normalized)}</code>`);
    });

    // Remote images are intentionally not embedded in this local-first UI.
    text = text.replace(/!\[([^\]\n]*)\]\(\s*([^\s)]+)(?:\s+(?:"[^"]*"|'[^']*'))?\s*\)/g,
      (_match, label, url) => {
        const destination = safeUrl(url);
        const alt = renderInline(label || "Hình ảnh");
        if (!destination) return stash(alt);
        return stash(`<a href="${escapeHtml(destination)}" target="_blank" rel="noopener noreferrer">${alt}</a>`);
      });

    text = text.replace(/\[([^\]\n]+)\]\(\s*([^\s)]+)(?:\s+(?:"([^"]*)"|'([^']*)'))?\s*\)/g,
      (match, label, url, doubleTitle, singleTitle) => {
        const destination = safeUrl(url);
        if (!destination) return stash(escapeHtml(match));
        const title = doubleTitle ?? singleTitle;
        const titleAttribute = title === undefined ? "" : ` title="${escapeHtml(title)}"`;
        return stash(
          `<a href="${escapeHtml(destination)}"${titleAttribute} target="_blank" rel="noopener noreferrer">${renderInline(label)}</a>`,
        );
      });

    text = escapeHtml(text);
    text = text.replace(/\*\*([^\n]+?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/__([^\n]+?)__/g, "<strong>$1</strong>");
    text = text.replace(/~~([^\n]+?)~~/g, "<del>$1</del>");
    text = text.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
    text = text.replace(/(^|[^\w])_([^_\n]+?)_(?!\w)/g, "$1<em>$2</em>");
    text = text.replace(/ {2,}\n/g, "<br>\n");

    for (let index = tokens.length - 1; index >= 0; index -= 1) {
      text = text.split(`${TOKEN_OPEN}${index}${TOKEN_CLOSE}`).join(tokens[index]);
    }
    return text;
  }

  function leadingSpaces(value) {
    return value.match(/^ */)[0].length;
  }

  function listItem(line) {
    const match = line.match(/^( *)([-+*]|\d+[.)])\s+(.+)$/);
    if (!match) return null;
    return {
      indent: match[1].length,
      ordered: /^\d/.test(match[2]),
      start: /^\d/.test(match[2]) ? Number.parseInt(match[2], 10) : null,
      text: match[3],
    };
  }

  function fence(line) {
    const match = line.match(/^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)\s*$/);
    return match ? { marker: match[1][0], length: match[1].length, language: match[2] } : null;
  }

  function isHorizontalRule(line) {
    const compact = line.trim().replace(/\s/g, "");
    return /^(?:\*{3,}|-{3,}|_{3,})$/.test(compact);
  }

  function splitTableRow(line) {
    let value = line.trim();
    if (value.startsWith("|")) value = value.slice(1);
    if (value.endsWith("|") && !value.endsWith("\\|")) value = value.slice(0, -1);

    const cells = [];
    let current = "";
    let escaped = false;
    let codeTicks = 0;
    for (let index = 0; index < value.length; index += 1) {
      const character = value[index];
      if (escaped) {
        current += character;
        escaped = false;
      } else if (character === "\\") {
        current += character;
        escaped = true;
      } else if (character === "`") {
        codeTicks = codeTicks ? 0 : 1;
        current += character;
      } else if (character === "|" && !codeTicks) {
        cells.push(current.trim());
        current = "";
      } else {
        current += character;
      }
    }
    cells.push(current.trim());
    return cells;
  }

  function tableAlignment(line) {
    if (!line.includes("|")) return null;
    const cells = splitTableRow(line);
    if (!cells.length || cells.some((cell) => !/^:?-{3,}:?$/.test(cell))) return null;
    return cells.map((cell) => {
      if (cell.startsWith(":") && cell.endsWith(":")) return "center";
      if (cell.endsWith(":")) return "right";
      return "left";
    });
  }

  function beginsBlock(lines, index) {
    const line = lines[index] ?? "";
    if (!line.trim()) return true;
    if (fence(line) || listItem(line) || isHorizontalRule(line)) return true;
    if (/^ {0,3}#{1,6}\s+/.test(line) || /^ {0,3}>/.test(line)) return true;
    if (/^ {4}\S/.test(line)) return true;
    return Boolean(index + 1 < lines.length && line.includes("|") && tableAlignment(lines[index + 1]));
  }

  function renderTable(lines, start) {
    const headers = splitTableRow(lines[start]);
    const alignments = tableAlignment(lines[start + 1]);
    let index = start + 2;
    const rows = [];
    while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
      rows.push(splitTableRow(lines[index]));
      index += 1;
    }

    const cell = (tag, value, column) => {
      const alignment = alignments[column] || "left";
      return `<${tag} class="align-${alignment}">${renderInline(value ?? "")}</${tag}>`;
    };
    const head = `<thead><tr>${headers.map((value, column) => cell("th", value, column)).join("")}</tr></thead>`;
    const body = rows.length
      ? `<tbody>${rows.map((row) => `<tr>${headers.map((_value, column) => cell("td", row[column], column)).join("")}</tr>`).join("")}</tbody>`
      : "";
    return { html: `<div class="table-scroll"><table>${head}${body}</table></div>`, index };
  }

  function renderList(lines, start) {
    const first = listItem(lines[start]);
    const tag = first.ordered ? "ol" : "ul";
    const startAttribute = first.ordered && first.start !== 1 ? ` start="${first.start}"` : "";
    const items = [];
    let index = start;

    while (index < lines.length) {
      const item = listItem(lines[index]);
      if (!item || item.indent !== first.indent || item.ordered !== first.ordered) break;

      const itemLines = [item.text];
      index += 1;
      while (index < lines.length) {
        const next = listItem(lines[index]);
        if (next && next.indent === first.indent) break;
        if (!lines[index].trim()) {
          let lookahead = index + 1;
          while (lookahead < lines.length && !lines[lookahead].trim()) lookahead += 1;
          const following = listItem(lines[lookahead] || "");
          if (following && following.indent === first.indent && following.ordered === first.ordered) {
            index = lookahead;
            break;
          }
          itemLines.push("");
          index += 1;
          continue;
        }
        if (!next && leadingSpaces(lines[index]) <= first.indent) break;
        const contentIndent = Math.min(lines[index].length, first.indent + 2);
        itemLines.push(lines[index].slice(contentIndent));
        index += 1;
      }

      let content = renderBlocks(itemLines).trim();
      if (/^<p>[\s\S]*<\/p>$/.test(content) && !content.slice(3, -4).includes("<p>")) {
        content = content.slice(3, -4);
      }
      items.push(`<li>${content}</li>`);
    }
    return { html: `<${tag}${startAttribute}>${items.join("")}</${tag}>`, index };
  }

  function renderBlocks(lines) {
    const blocks = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      const codeFence = fence(line);
      if (codeFence) {
        index += 1;
        const code = [];
        const closing = new RegExp(`^ {0,3}${codeFence.marker}{${codeFence.length},}\\s*$`);
        while (index < lines.length && !closing.test(lines[index])) {
          code.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        const language = /^[\w+-]{1,32}$/.test(codeFence.language)
          ? ` class="language-${escapeHtml(codeFence.language)}"`
          : "";
        blocks.push(`<pre><code${language}>${escapeHtml(code.join("\n"))}</code></pre>`);
        continue;
      }

      const heading = line.match(/^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
      if (heading) {
        const level = heading[1].length;
        blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        index += 1;
        continue;
      }

      if (isHorizontalRule(line)) {
        blocks.push("<hr>");
        index += 1;
        continue;
      }

      if (/^ {0,3}>/.test(line)) {
        const quote = [];
        while (index < lines.length && (/^ {0,3}>/.test(lines[index]) || !lines[index].trim())) {
          quote.push(lines[index].replace(/^ {0,3}> ?/, ""));
          index += 1;
        }
        blocks.push(`<blockquote>${renderBlocks(quote)}</blockquote>`);
        continue;
      }

      if (/^ {4}\S/.test(line)) {
        const code = [];
        while (index < lines.length && (/^ {4}/.test(lines[index]) || !lines[index].trim())) {
          code.push(lines[index].replace(/^ {4}/, ""));
          index += 1;
        }
        while (code.length && !code[code.length - 1]) code.pop();
        blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        continue;
      }

      const alignments = index + 1 < lines.length ? tableAlignment(lines[index + 1]) : null;
      if (line.includes("|") && alignments) {
        const table = renderTable(lines, index);
        blocks.push(table.html);
        index = table.index;
        continue;
      }

      if (listItem(line)) {
        const list = renderList(lines, index);
        blocks.push(list.html);
        index = list.index;
        continue;
      }

      const paragraph = [line.trim()];
      index += 1;
      while (index < lines.length && lines[index].trim() && !beginsBlock(lines, index)) {
        paragraph.push(lines[index].trim());
        index += 1;
      }
      blocks.push(`<p>${renderInline(paragraph.join("\n"))}</p>`);
    }
    return blocks.join("");
  }

  function render(markdown) {
    const normalized = String(markdown ?? "").replace(/\r\n?/g, "\n");
    return renderBlocks(normalized.split("\n"));
  }

  return Object.freeze({ render });
});
