"""Gemini/Generative API integration helper

This module calls an external generative model (Gemini/Text-Bison or custom) to
produce personalized healthcare tips based on user context and recent messages.

Configuration (environment variables):
- GEMINI_API_KEY: Bearer token for the generative API
- GEMINI_API_URL: Full URL to the generate endpoint. Defaults to
  https://generativelanguage.googleapis.com/v1/models/text-bison-001:generate

Note: Do NOT store secrets in source. Set GEMINI_API_KEY in the deployment environment.
"""
import os
import json
import httpx
import asyncio
from typing import List, Dict, Any

DEFAULT_API_URL = os.getenv(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1/models/text-bison-001:generate",
)

API_KEY = os.getenv("GEMINI_API_KEY")

async def generate_tips_via_gemini(user_context: Dict[str, Any], messages: List[Dict[str, Any]], max_output_tokens: int = 250) -> Dict[str, Any]:
    """Call the configured generative API to produce personalized tips.

    Returns a dict: { "tips": [...], "raw": "full model text", "urgency": "low|medium|high" }
    """
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured in environment")

    # Build a compact prompt that instructs the model to output JSON
    prompt_lines = []
    prompt_lines.append("You are a professional, empathetic healthcare assistant.")
    prompt_lines.append("Based on the provided user context and recent chat messages, produce up to 5 concise, prioritized, evidence-based healthcare tips tailored to the user's mood and symptoms. Return the output strictly as a JSON object with keys: tips (array of strings), urgency (low|medium|high). Do NOT include any extra commentary outside JSON.")

    # Add context
    mood_emoji = user_context.get("moodEmoji") or user_context.get("mood") or ""
    mood_name = user_context.get("mood", {}).get("name") if isinstance(user_context.get("mood"), dict) else user_context.get("mood_state") or "unknown"
    prompt_lines.append(f"Mood: {mood_emoji} ({mood_name})")

    symptoms = user_context.get("symptoms", [])
    if symptoms:
        prompt_lines.append("Symptoms: " + ", ".join(symptoms))

    topics = user_context.get("topics") or {}
    if isinstance(topics, dict) and topics:
        top_keys = ", ".join(sorted(topics.keys(), key=lambda k: -topics[k]))
        prompt_lines.append("Top topics: " + top_keys)

    # Add recent messages
    recent_texts = []
    for m in messages[-10:]:
        who = m.get("type") or m.get("role") or "user"
        txt = (m.get("text") or m.get("content") or "").strip()
        if txt:
            recent_texts.append(f"{who}: {txt}")
    if recent_texts:
        prompt_lines.append("Recent messages:\n" + "\n".join(recent_texts))

    # Add short instruction about tone and format
    prompt_lines.append("Tone: empathetic, concise. Output format: JSON only. Max 5 tips.")

    prompt = "\n\n".join(prompt_lines)

    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.2,
        "maxOutputTokens": max_output_tokens
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(DEFAULT_API_URL, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except Exception as e:
            # Bubble up error message
            raise RuntimeError(f"Generative API request failed: {e} - {resp.text}")

        data = resp.json()

    # Attempt to extract text output (supporting several response shapes)
    raw_text = ""
    # Google generative API (v1) often returns 'candidates' -> [{'output': '...'}] or 'candidates'->'content'
    if isinstance(data, dict):
        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
            cand = data["candidates"][0]
            raw_text = cand.get("output") or cand.get("content") or cand.get("text") or json.dumps(cand)
        elif "output" in data and isinstance(data["output"], list) and data["output"]:
            raw_text = data["output"][0].get("content", "")
        elif "text" in data:
            raw_text = data.get("text")
        else:
            # Fallback: stringify entire response
            raw_text = json.dumps(data)
    else:
        raw_text = str(data)

    # Try to parse JSON from raw_text
    tips = []
    urgency = "low"
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            tips = parsed.get("tips") or parsed.get("suggestions") or []
            urgency = parsed.get("urgency") or parsed.get("priority") or urgency
    except Exception:
        # Raw text was not JSON; fall back to naive extraction
        lines = [ln.strip().lstrip("-\u2022 ") for ln in raw_text.splitlines() if ln.strip()]
        # Select lines that are likely tips (start with digit or bullet or are short sentences)
        extracted = []
        for ln in lines:
            if len(extracted) >= 5:
                break
            # Skip header lines
            if ln.lower().startswith("mood:") or ln.lower().startswith("symptoms:") or ln.lower().startswith("tone:"):
                continue
            extracted.append(ln)
        tips = extracted

    # Normalize tips
    tips_clean = []
    for t in tips:
        if isinstance(t, str):
            s = t.strip()
            if s and s not in tips_clean:
                tips_clean.append(s)

    return {"tips": tips_clean, "raw": raw_text, "urgency": urgency}
