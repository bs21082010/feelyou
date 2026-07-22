import os
import json
import hashlib
import urllib.request
import urllib.parse
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:11434/api/chat")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

PERSONAS = {
    "mentor": "You are a warm, empathetic mentor. Always respond with encouragement, positivity, and emotional awareness. Use supportive language, motivational tone, and show understanding. Never give harmful or misleading advice. If uncertain, say you don't know but offer to help explore it.",
    "coach": "You are a direct, results-oriented coach. Always respond with clarity, actionable steps, and accountability. Push the user toward growth with firm but supportive language. Never give harmful or misleading advice. If uncertain, say you don't know but offer to help explore it.",
    "teacher": "You are a patient, knowledgeable teacher. Always respond with structured explanations, examples, and clarity. Break complex topics into digestible steps. Never give harmful or misleading advice. If uncertain, say you don't know but offer to help explore it.",
}

FALLBACK = "I don't have that information right now, but I can help you explore it."
PROMPT_TEMPLATES = {
    "explain": "Explain this in a clear, kind way:\n\n{{query}}",
    "motivate": "Give me a motivational push about:\n\n{{query}}",
    "advise": "Offer supportive advice on:\n\n{{query}}",
}
RECENT_SCORES: list = []
CONSECUTIVE_LOW = 0
LAST_REPLY = None
LAST_PROMPT = None
LAST_SCORE = None

DRIFT_THRESHOLD = 15
SCORE_THRESHOLD = 70
WEIGHT_BOOST_FACTOR = 12
WEIGHT_DECAY_PER_TURN = 3
PREDICTIVE_BOOST = 8
CONFLICT_AUTO_POLICY = "low"

ESCAPED_MODEL = re.sub(r'[^a-zA-Z0-9_-]', '_', LLM_MODEL)


class ChatRequest(BaseModel):
    message: str
    persona: str = "mentor"
    weights: Optional[dict] = None
    conversation: Optional[list] = None


class ChatResponse(BaseModel):
    reply: str
    score: int
    weights: dict
    intent: Optional[str] = None
    escalation_tier: Optional[int] = None


def score_reply(reply: str) -> int:
    score = 100
    words = reply.split()
    if len(words) < 20: score -= 20
    if len(words) < 10: score -= 15
    lower = reply.lower()
    if "i don't know" in lower or "unsure" in lower: score -= 30
    if "i don't have that information" in lower: score -= 10
    if any(w in lower for w in ["maybe", "sort of", "perhaps", "kind of"]): score -= 10
    if reply.endswith("?") or reply.endswith("..."): score -= 5
    return max(score, 0)


def detect_intent(text: str) -> Optional[str]:
    lower = text.lower()
    if any(w in lower for w in ["motivate", "inspire", "encourage", "push", "hype"]): return "motivate"
    if any(w in lower for w in ["explain", "what is", "how does", "define", "tell me about"]): return "explain"
    if any(w in lower for w in ["advise", "should i", "what should", "recommend", "suggest"]): return "advise"
    return None


def normalize_weights(w: dict):
    if not w: return
    total = sum(w.values())
    if total == 0: return
    factor = 100 / total
    for k in w: w[k] = round(w[k] * factor)


def decay_weights(w: dict):
    for k in w:
        if w[k] > 50: w[k] = max(50, w[k] - WEIGHT_DECAY_PER_TURN)


def predictive_intent_boost(intent: str) -> dict:
    return {}


def apply_template(user_input: str, dynamic_weights: dict = None) -> str:
    intent = detect_intent(user_input) if dynamic_weights is not None else None
    if intent and dynamic_weights is not None:
        decay_weights(dynamic_weights)
        if intent == "motivate":
            for n in ["coach", "mentor"]:
                if n in dynamic_weights: dynamic_weights[n] = min(dynamic_weights.get(n, 50) + WEIGHT_BOOST_FACTOR, 100)
        elif intent == "explain" and "teacher" in dynamic_weights:
            dynamic_weights["teacher"] = min(dynamic_weights["teacher"] + WEIGHT_BOOST_FACTOR, 100)
        elif intent == "advise":
            for n in ["mentor", "coach"]:
                if n in dynamic_weights: dynamic_weights[n] = min(dynamic_weights.get(n, 50) + int(WEIGHT_BOOST_FACTOR * 0.8), 100)
        normalize_weights(dynamic_weights)
    for key, template in PROMPT_TEMPLATES.items():
        if user_input.lower().startswith(key):
            return template.format(query=user_input[len(key):].strip())
    return user_input


def build_persona(persona_name: str, weights: dict = None) -> str:
    if "+" in persona_name:
        parts = persona_name.split("+")
        segs = []
        for part in parts:
            part = part.strip()
            if ":" in part:
                name, ws = part.rsplit(":", 1)
                weight = int(ws) if ws.isdigit() else 50
            else:
                name = part
                weight = 50
            base = PERSONAS.get(name.strip())
            if weights and name.strip() in weights:
                weight = weights[name.strip()]
            if base:
                segs.append((base.strip(), weight))
        if not segs:
            return PERSONAS["mentor"]
        tw = sum(w for _, w in segs) or 1
        parts_out = []
        for t, w in segs:
            pct = w / tw
            emp = "strongly" if pct > 0.5 else "moderately" if pct > 0.2 else "slightly"
            parts_out.append(f"[{emp} weighted at {w}%]\n{t}")
        persona = "\n\n".join(parts_out)
    else:
        persona = PERSONAS.get(persona_name, PERSONAS["mentor"])
    avg_c = (sum(RECENT_SCORES) / len(RECENT_SCORES)) if RECENT_SCORES else 100
    if avg_c < 50: persona += "\nBe cautious and acknowledge uncertainty clearly."
    elif avg_c < 70: persona += "\nBalance confidence with openness to correction."
    else: persona += "\nRespond with strong confidence and clarity."
    return persona


def call_llm(system_prompt: str, user_message: str) -> str:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    body = json.dumps({
        "model": LLM_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }).encode()
    req = urllib.request.Request(LLM_API_URL, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "") or data.get("choices", [{}])[0].get("message", {}).get("content", "") or FALLBACK
    except Exception as e:
        return FALLBACK + f" (Error: {str(e)[:60]})"


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    global RECENT_SCORES, CONSECUTIVE_LOW, LAST_REPLY, LAST_PROMPT, LAST_SCORE

    weights = req.weights or {}
    prompt = apply_template(req.message, weights)
    persona = build_persona(req.persona, weights)

    reply = call_llm(persona, prompt)
    score = score_reply(reply)
    RECENT_SCORES.append(score)
    if len(RECENT_SCORES) > 20: RECENT_SCORES.pop(0)
    CONSECUTIVE_LOW = CONSECUTIVE_LOW + 1 if score < 50 else 0

    LAST_PROMPT = prompt if LAST_PROMPT is None else LAST_PROMPT
    LAST_REPLY = reply if LAST_REPLY is None else LAST_REPLY
    LAST_SCORE = score if LAST_SCORE is None else LAST_SCORE

    intent = detect_intent(req.message)
    escalation_tier = None
    if score < 50 and CONSECUTIVE_LOW >= 3:
        if CONSECUTIVE_LOW >= 7: escalation_tier = 3
        elif CONSECUTIVE_LOW >= 5: escalation_tier = 2
        else: escalation_tier = 1

    if weights:
        normalize_weights(weights)

    return ChatResponse(
        reply=reply,
        score=score,
        weights=weights,
        intent=intent,
        escalation_tier=escalation_tier,
    )


@app.get("/api/dashboard")
async def dashboard_endpoint():
    return {
        "recent_scores": RECENT_SCORES,
        "consecutive_low": CONSECUTIVE_LOW,
        "avg_confidence": round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None,
        "drift_threshold": DRIFT_THRESHOLD,
        "score_threshold": SCORE_THRESHOLD,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL, "llm_url": LLM_API_URL}
