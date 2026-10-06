---
name: ally
description: >
  Ally parses images and document screenshots, extracting text,
  tables, chart data, and layout structure. Use Ally when the user
  uploads an image, screenshot, or needs information extracted from
  visual content. Supports OCR, table structuring, chart description,
  and document summarization. Trigger keywords: image, screenshot,
  photo, scan, table, chart, receipt, invoice, extract text, read
  this, what does this say.
license: Apache-2.0
compatibility: Requires Python 3.11+ and access to a vision-language model endpoint (local Ollama or OpenAI-compatible API)
metadata:
  author: your-team
  version: "1.0"
  homepage: https://github.com/YOUR-USERNAME/ally
---

# Ally — Multimodal Document Analyst

## When to use Ally

Invoke Ally when the user:
- Uploads an image, screenshot, photo, or scanned document.
- Asks you to "read", "extract", "transcribe", or "summarize" visual content.
- Shares a table, chart, receipt, invoice, form, or diagram as an image.
- Refers to visual content that you cannot interpret as plain text.

Do **not** invoke Ally for:
- Plain text files (`.txt`, `.md`, `.csv`) — read them directly.
- PDFs with an embedded text layer — extract text first, use Ally only for image-only pages.
- General image generation or editing tasks.

## How Ally works

Ally runs a four-step pipeline. Follow these steps in order.

### Step 1 — Preprocess the image

Run `scripts/preprocess.py` on the input. It will:
- Downscale images larger than 2048px on the long edge.
- Detect document-like regions and split them into overlapping tiles
  if the source is very wide or very tall (e.g., long receipts).
- Return a list of image paths plus metadata.

Do not skip this step — feeding a 6000px screenshot directly to a
2B-parameter model degrades OCR accuracy noticeably.

### Step 2 — Run multimodal inference

Call `scripts/analyze.py` with the preprocessed image and a task
template. Supported tasks:

| Task | Purpose |
|---|---|
| `ocr_extract` | Plain text extraction, preserving reading order. |
| `table_structuring` | Convert a table image to JSON headers + rows. |
| `chart_summary` | Describe chart type, axes, series, and key values. |
| `doc_summary` | Condense multi-image or multi-page documents. |
| `layout_parse` | Return bounding regions for forms and receipts. |

Always use a task template. Free-form prompts produce unstable output
that is hard to parse downstream.

### Step 3 — Normalize output

Ally returns a JSON object with a fixed schema. Always validate the
response against this shape before passing it to the host Agent:

```json
{
  "type": "table",
  "confidence": 0.92,
  "content": { "...": "..." },
  "notes": "Optional caveats, assumptions, or warnings."
}