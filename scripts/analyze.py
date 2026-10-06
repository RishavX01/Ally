#!/usr/bin/env python3
"""Ally multimodal inference: call the vision backend and normalize output."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

TASK_TEMPLATES = {
    "ocr_extract": (
        "Extract all visible text from this image, preserving reading "
        "order. Return JSON: {{\"type\":\"text\",\"confidence\":<0-1>,"
        "\"content\":{{\"text\":\"<extracted>\"}},\"notes\":\"\"}}"
    ),
    "table_structuring": (
        "This image contains a table. Extract it as JSON: "
        "{{\"type\":\"table\",\"confidence\":<0-1>,\"content\":"
        "{{\"headers\":[...],\"rows\":[[...],...]}},\"notes\":\"\"}}"
    ),
    "chart_summary": (
        "Describe this chart. Return JSON: {{\"type\":\"chart\","
        "\"confidence\":<0-1>,\"content\":{{\"chart_type\":\"\","
        "\"axes\":{{\"x\":\"\",\"y\":\"\"}},\"series\":[],"
        "\"key_values\":[]}},\"notes\":\"\"}}"
    ),
    "doc_summary": (
        "Summarize this document image. Return JSON: {{\"type\":\"summary\","
        "\"confidence\":<0-1>,\"content\":{{\"summary\":\"\","
        "\"key_points\":[]}},\"notes\":\"\"}}"
    ),
    "layout_parse": (
        "Identify layout regions (text blocks, tables, images) with "
        "approximate bounding boxes. Return JSON: {{\"type\":\"layout\","
        "\"confidence\":<0-1>,\"content\":{{\"regions\":[{{\"kind\":\"\","
        "\"bbox\":[x,y,w,h],\"text\":\"\"}}]}},\"notes\":\"\"}}"
    ),
}


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_ollama(image_path: Path, prompt: str, model: str,
                base: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_encode(image_path)],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    r = requests.post(f"{base}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")


def call_openai_compatible(image_path: Path, prompt: str, model: str,
                           base: str, api_key: str,
                           max_tokens: int) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_encode(image_path)}"
                }},
            ],
        }],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(f"{base}/v1/chat/completions",
                      json=payload, headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_json(raw: str) -> dict | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def normalize(raw: str, task: str) -> dict:
    parsed = extract_json(raw)
    if parsed is None:
        return {
            "type": "raw",
            "confidence": 0.0,
            "content": {"text": raw.strip()},
            "notes": f"Model did not return valid JSON for task '{task}'.",
        }
    parsed.setdefault("type", task)
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("content", {})
    parsed.setdefault("notes", "")
    return parsed


def run(image_path: Path, task: str, lang: str | None = None) -> dict:
    if task not in TASK_TEMPLATES:
        raise SystemExit(f"unknown task: {task}. "
                         f"Choose from {list(TASK_TEMPLATES)}")

    backend = os.getenv("ALLY_BACKEND", "ollama")
    model = os.getenv("ALLY_MODEL", "qwen3-vl:4b")
    base = os.getenv("ALLY_API_BASE", "http://localhost:11434")
    api_key = os.getenv("ALLY_API_KEY", "")
    max_tokens = int(os.getenv("ALLY_MAX_TOKENS", "1120"))

    prompt = TASK_TEMPLATES[task]
    if lang:
        prompt = f"Content language: {lang}. " + prompt

    if backend == "ollama":
        raw = call_ollama(image_path, prompt, model, base, max_tokens)
    elif backend == "openai_compatible":
        raw = call_openai_compatible(image_path, prompt, model, base,
                                     api_key, max_tokens)
    else:
        raise SystemExit(f"unknown backend: {backend}")

    return normalize(raw, task)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ally vision inference")
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--task", required=True, choices=list(TASK_TEMPLATES))
    ap.add_argument("--lang", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1

    result = run(args.image, args.task, args.lang)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
