"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const markdown = require("../rag_app/static/markdown.js");

test("renders headings, emphasis, lists, and citations", () => {
  const html = markdown.render([
    "## Tín hiệu tích cực",
    "",
    "- **Đảo chiều:** Giá vượt EMA 10 [1], [3].",
    "- *RSI* hồi phục.",
  ].join("\n"));

  assert.match(html, /<h2>Tín hiệu tích cực<\/h2>/);
  assert.match(html, /<ul><li><strong>Đảo chiều:<\/strong> Giá vượt EMA 10 \[1\], \[3\]\.<\/li>/);
  assert.match(html, /<li><em>RSI<\/em> hồi phục\.<\/li><\/ul>/);
});

test("renders fenced code and GitHub-style tables", () => {
  const html = markdown.render([
    "| Mã | RSI |",
    "| :--- | ---: |",
    "| SSI | 55 |",
    "",
    "```text",
    "<not-html>",
    "```",
  ].join("\n"));

  assert.match(html, /<th class="align-left">Mã<\/th>/);
  assert.match(html, /<td class="align-right">55<\/td>/);
  assert.match(html, /<pre><code class="language-text">&lt;not-html&gt;<\/code><\/pre>/);
});

test("escapes raw HTML and rejects unsafe links", () => {
  const html = markdown.render([
    '<img src=x onerror="alert(1)">',
    "",
    "[không an toàn](javascript:alert(1))",
    "",
    "[nguồn](https://example.com)",
  ].join("\n"));

  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
  assert.doesNotMatch(html, /href="javascript:/);
  assert.match(html, /href="https:\/\/example\.com"/);
  assert.match(html, /rel="noopener noreferrer"/);
});
