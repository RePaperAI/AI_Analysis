"""Optional Gemini image review for RePaper AI.

This module never controls the printer and is disabled unless GEMINI_API_KEY
is present in the environment.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request


def review_sheet(image_bytes: bytes, mime_type: str = "image/jpeg") -> str | None:
    """Ask Gemini for a short visual review, or return None when disabled."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    prompt = (
        "Review this paper scan for a recycling/cutting workflow. Identify visible "
        "writing, printed regions, objects, glare, blur, clipped edges, or other "
        "areas that should not be cut. Give concise practical advice. Do not invent "
        "measurements and do not control any machine."
    )
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
        ]}],
        "generationConfig": {"temperature": 0.1},
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Gemini review failed: {exc}") from exc
    parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    return text.strip() or None
