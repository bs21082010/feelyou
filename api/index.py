import os
import json
import hashlib
import urllib.request
import urllib.parse
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
TRAINING_EVENTS: list = []
LAST_TRAINING_NARRATION = None
RECENT_SCORES: list = []
CONSECUTIVE_LOW = 0
LAST_REPLY = None
LAST_PROMPT = None
LAST_SCORE = None
WEAK_REPLY_COUNT = 0
LAST_SYNC_TIME = None
ESCALATION_EVENTS: list = []
LAST_SYNC_STATUS = "never"
CONFLICT_RESOLUTIONS: list = []
HOTSPOT_REINFORCED: set = set()
WEAK_INTENT_COUNTER: dict = {}
SYNC_COUNT = 0
SYNC_HISTORY_BY_USER: dict = defaultdict(list)
USER_CONFLICTS: dict = defaultdict(list)
CONTRIBUTORS: dict = {}
FEWSHOT_DATASET: list = []
AUTO_REINFORCED_INTENTS: set = set()
ADAPTIVE_HISTORY: list = []
FEWSHOT_PATH = os.path.join(HERE, "fewshot_dataset.jsonl")
USER_WEAK_REPLIES: dict = defaultdict(int)
USER_GOLD_EXPORTS: dict = defaultdict(int)
USER_ANALYTICS_HISTORY: dict = defaultdict(list)

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


MOCK_REPLIES = {
    "motivate": [
        "You've got this! Every step forward, no matter how small, builds momentum. The fact that you're here asking shows you're already committed to growth. Keep your eyes on the goal, break it down into daily actions, and celebrate each win along the way. I believe in you!",
        "I hear you, and it's okay to feel the way you do. What matters is that you're still showing up. Let's channel that energy into one small, concrete action you can take right now. Progress isn't about giant leaps — it's about consistent steps.",
        "The very act of reaching out tells me you have the drive to move forward. Motivation isn't a feeling you wait for — it's a muscle you build. Let's start with something small and build from there. You're stronger than you think.",
    ],
    "explain": [
        "Great question! Let me break this down simply. Think of it like building blocks — each concept builds on the previous one. Start with the core idea, understand how the pieces connect, and before you know it, the bigger picture becomes clear. Would you like me to go deeper on any specific part?",
        "That's a really good question. Here's the simplest way to think about it: imagine you're learning a new language. You don't start with complex sentences — you start with single words, then phrases, then full conversations. Same idea here. Let me walk you through it step by step.",
        "I love questions like this. The key insight is actually pretty straightforward once you strip away the jargon. At its core, this is about connecting two ideas: cause and effect. Here's a real-world example that makes it click.",
    ],
    "advise": [
        "Here's my take: take a step back and look at what's really important here. Focus on what you can control, set clear boundaries, and remember that progress matters more than perfection. I'd suggest starting with one small actionable step today. What does that look like for you?",
        "That's a tough spot to be in, and I respect you for working through it. Let me offer a different lens: instead of asking 'what's the right choice,' ask 'which option teaches me more, regardless of the outcome?' Growth often hides in the harder path.",
        "I've seen this pattern before. The best approach is usually to separate what you can control from what you can't. Make a quick list of both. Then put your energy entirely into the things you can influence. You'd be surprised how much clarity that brings.",
    ],
}
MOCK_GENERAL = [
    "That's an interesting point. Here's what I think — every challenge carries a lesson, and every question opens a door. Keep exploring, keep asking, and trust the process.",
    "I appreciate you sharing that. The best insights often come from honest conversations. Let's sit with that thought for a moment and see where it leads us.",
    "Thanks for bringing that up. I'd say the key is to stay curious and keep an open mind. There's always more to discover, and you're on the right track by engaging with these ideas.",
    "That's a great perspective. Remember that growth isn't always linear — some days feel like breakthroughs and others feel like setbacks, but it's all part of the journey.",
    "That's a thoughtful question. The answer often depends on your specific context, but here's a principle that usually applies: start simple, iterate fast, and learn from each attempt.",
    "I'm glad you asked. This is one of those topics where the journey matters as much as the destination. Let's explore it together and see what resonates with you.",
    "There's a lot to unpack here, but I think the heart of it is simpler than it seems. Let's focus on the core idea and work outward from there.",
    "That's a valuable insight you're touching on. The fact that you're reflecting on this shows a lot of self-awareness. Let's build on that.",
    "I think what you're really asking is deeper than it appears on the surface. Let me reframe it a bit and see if we can find the underlying thread together.",
    "That resonates with a lot of people. You're not alone in feeling this way. The key isn't to have all the answers — it's to ask better questions.",
]

import random

def generate_mock_reply(user_message: str, persona_name: str) -> str:
    intent = detect_intent(user_message)
    if intent and intent in MOCK_REPLIES:
        base = random.choice(MOCK_REPLIES[intent])
    else:
        base = random.choice(MOCK_GENERAL)
    if "coach" in persona_name:
        base += " Now, what's your next step going to be?"
    elif "teacher" in persona_name:
        base += " Does that clarify things for you?"
    else:
        base += " How does that resonate with you?"
    return base


def call_llm(system_prompt: str, user_message: str, persona_name: str = "mentor") -> str:
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
        mock = generate_mock_reply(user_message, persona_name)
        return mock


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    global RECENT_SCORES, CONSECUTIVE_LOW, LAST_REPLY, LAST_PROMPT, LAST_SCORE, WEAK_REPLY_COUNT, ESCALATION_EVENTS
    user_id = request.headers.get("X-User-Id") or "anonymous"

    weights = req.weights or {}
    prompt = apply_template(req.message, weights)
    persona = build_persona(req.persona, weights)

    reply = call_llm(persona, prompt, req.persona)
    score = score_reply(reply)
    RECENT_SCORES.append(score)
    if len(RECENT_SCORES) > 20: RECENT_SCORES.pop(0)
    CONSECUTIVE_LOW = CONSECUTIVE_LOW + 1 if score < 50 else 0

    if score < 50:
        WEAK_REPLY_COUNT += 1
        USER_WEAK_REPLIES[user_id] += 1
        TRAINING_EVENTS.append({
            "type": "weak_reply",
            "score": score,
            "intent": intent or "general",
            "timestamp": datetime.now().isoformat(),
            "weak_count": WEAK_REPLY_COUNT,
        })
    LAST_REPLY = reply if LAST_REPLY is None else LAST_REPLY
    LAST_SCORE = score if LAST_SCORE is None else LAST_SCORE

    intent = detect_intent(req.message)
    escalation_tier = None
    if score < 50 and CONSECUTIVE_LOW >= 3:
        if CONSECUTIVE_LOW >= 7: escalation_tier = 3
        elif CONSECUTIVE_LOW >= 5: escalation_tier = 2
        else: escalation_tier = 1
        ESCALATION_EVENTS.append({
            "tier": escalation_tier,
            "intent": intent or "general",
            "timestamp": datetime.now().isoformat(),
        })
        intent_key = intent or "general"
        WEAK_INTENT_COUNTER[intent_key] = WEAK_INTENT_COUNTER.get(intent_key, 0) + 1

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
    return {
        "status": "ok",
        "model": LLM_MODEL,
        "llm_url": LLM_API_URL,
        "has_api_key": bool(LLM_API_KEY),
        "mode": "offline" if "localhost" in LLM_API_URL and not LLM_API_KEY else "configured",
    }


@app.get("/api/heatmap-narration")
async def heatmap_narration():
    if not ESCALATION_EVENTS:
        return {"narration": "No escalation events recorded yet."}
    tiers = Counter(e["tier"] for e in ESCALATION_EVENTS)
    intents = Counter(e["intent"] for e in ESCALATION_EVENTS)
    total = len(ESCALATION_EVENTS)
    most_common_intent = intents.most_common(1)
    parts = [f"{total} escalation events total."]
    for t in sorted(tiers):
        label = {1: "external lookups", 2: "persona shifts", 3: "hard fallbacks"}.get(t, f"tier {t}")
        parts.append(f"{tiers[t]} {label}.")
    if most_common_intent:
        parts.append(f"Most frequent trigger is {most_common_intent[0][0]} with {most_common_intent[0][1]} events.")
    return {"narration": " ".join(parts)}


@app.get("/api/sync-status")
async def sync_status():
    return {
        "last_sync": LAST_SYNC_STATUS,
        "last_sync_time": LAST_SYNC_TIME,
        "weak_reply_count": WEAK_REPLY_COUNT,
        "contributors": list(CONTRIBUTORS.values()),
        "total_contributors": len(CONTRIBUTORS),
    }


@app.post("/api/sync")
async def trigger_sync(req: Request):
    global LAST_SYNC_TIME, LAST_SYNC_STATUS, SYNC_COUNT
    user_id = req.headers.get("X-User-Id") or "anonymous"
    SYNC_COUNT += 1
    LAST_SYNC_TIME = datetime.now().isoformat()
    LAST_SYNC_STATUS = "synced"
    record = {"sync_number": SYNC_COUNT, "timestamp": LAST_SYNC_TIME, "user_id": user_id}
    SYNC_HISTORY_BY_USER[user_id].append(record)
    CONTRIBUTORS[user_id] = {
        "user_id": user_id,
        "last_sync": LAST_SYNC_TIME,
        "sync_count": len(SYNC_HISTORY_BY_USER[user_id]),
        "conflict_count": len(USER_CONFLICTS[user_id]),
        "status": "active",
    }
    return {
        "status": "ok",
        "synced_at": LAST_SYNC_TIME,
        "sync_count": SYNC_COUNT,
        "total_scores": len(RECENT_SCORES),
        "weak_count": WEAK_REPLY_COUNT,
        "avg_confidence": round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None,
        "user_id": user_id,
        "contributor": CONTRIBUTORS[user_id],
    }


@app.get("/api/weak-alert")
async def weak_alert():
    threshold = 3
    return {
        "count": WEAK_REPLY_COUNT,
        "alert": WEAK_REPLY_COUNT >= threshold,
        "threshold": threshold,
        "message": f"⚠️ {WEAK_REPLY_COUNT} weak repl{'y' if WEAK_REPLY_COUNT == 1 else 'ies'} detected. {'Consider retraining or adjusting persona.' if WEAK_REPLY_COUNT >= threshold else 'All good.'}" if WEAK_REPLY_COUNT >= threshold else None,
    }


@app.get("/api/training-narration")
async def training_narration():
    global LAST_TRAINING_NARRATION
    if not TRAINING_EVENTS:
        return {"narration": "No training events recorded yet.", "events": []}
    weak = [e for e in TRAINING_EVENTS if e["type"] == "weak_reply"]
    reinforces = [e for e in TRAINING_EVENTS if e["type"] == "reinforce"]
    recent = TRAINING_EVENTS[-20:]
    since_sync = [e for e in recent if LAST_SYNC_TIME is None or e.get("timestamp", "") >= LAST_SYNC_TIME]
    parts = []
    if weak:
        recent_weak = [e for e in weak if LAST_SYNC_TIME is None or e.get("timestamp", "") >= LAST_SYNC_TIME]
        if recent_weak:
            parts.append(f"Since last sync: {len(recent_weak)} new weak replies. Most recent score: {recent_weak[-1]['score']} for intent '{recent_weak[-1]['intent']}'.")
        else:
            parts.append(f"No new weak replies since last sync. Total weak replies tracked: {len(weak)}.")
    if reinforces:
        recent_force = [e for e in reinforces if LAST_SYNC_TIME is None or e.get("timestamp", "") >= LAST_SYNC_TIME]
        if recent_force:
            parts.append(f"Since last sync: {len(recent_force)} reinforcements applied. Last reinforced intent: {recent_force[-1]['intent']}.")
        else:
            parts.append(f"Reinforcements applied historically: {len(reinforces)}. Last reinforced: {reinforces[-1]['intent']}.")
    if not parts:
        parts.append("No significant training events since last sync.")
    parts.append(f"Total training events tracked: {len(TRAINING_EVENTS)}.")
    narration = " ".join(parts)
    LAST_TRAINING_NARRATION = narration
    return {"narration": narration, "events": recent}


@app.get("/api/training-history")
async def training_history():
    return {
        "total_events": len(TRAINING_EVENTS),
        "weak_count": len([e for e in TRAINING_EVENTS if e["type"] == "weak_reply"]),
        "reinforce_count": len([e for e in TRAINING_EVENTS if e["type"] == "reinforce"]),
        "recent": TRAINING_EVENTS[-20:],
    }


@app.post("/api/reset-training")
async def reset_training():
    global TRAINING_EVENTS, LAST_TRAINING_NARRATION, WEAK_REPLY_COUNT
    TRAINING_EVENTS.clear()
    LAST_TRAINING_NARRATION = None
    WEAK_REPLY_COUNT = 0
    return {"status": "ok", "cleared": True}


@app.get("/api/reinforce")
async def reinforce_hotspots():
    global HOTSPOT_REINFORCED
    if not WEAK_INTENT_COUNTER:
        return {"reinforced": False, "message": "No weak intents to reinforce."}
    riskiest = max(WEAK_INTENT_COUNTER, key=WEAK_INTENT_COUNTER.get)
    count = WEAK_INTENT_COUNTER[riskiest]
    if count < 2 or riskiest in HOTSPOT_REINFORCED:
        return {"reinforced": False, "message": f"Hotspot \"{riskiest}\" already reinforced or below threshold ({count})."}
    HOTSPOT_REINFORCED.add(riskiest)
    reinforcements = {
        "motivate": "Motivate me to start a new habit. Respond with warmth and actionable steps.",
        "explain": "Explain how to stay consistent with daily goals. Be clear and structured.",
        "advise": "What should I do when I feel overwhelmed? Offer supportive advice.",
    }
    template = reinforcements.get(riskiest, f"Help me with {riskiest}. Be supportive and clear.")
    reinforced = {
        "intent": riskiest,
        "count": count,
        "template": template,
    }
    TRAINING_EVENTS.append({
        "type": "reinforce",
        "intent": riskiest,
        "count": count,
        "timestamp": datetime.now().isoformat(),
    })
    return {"reinforced": True, "message": f"Reinforced \"{riskiest}\" with few-shot example ({count} escalations).", "detail": reinforced}


FEWSHOT_TEMPLATES = {
    "motivate": {
        "prompt": "I'm feeling really down and need motivation to keep going.",
        "completion": "I hear you. It's completely okay to have days where you feel low — that's part of being human. What matters is that you're still here, still trying, and that takes real courage. Let's take one small step together. What's one thing you can do in the next five minutes that might bring you a tiny bit of peace or progress? You've got this, and I'm right here with you.",
    },
    "explain": {
        "prompt": "Can you explain this to me like I'm five years old?",
        "completion": "Of course! Let me start with the simplest way to think about it. Imagine you're building with LEGO blocks — each new piece connects to something you already understand. The core idea is really just one small concept, and once that clicks, everything else builds on it naturally. Here's the simplest way to look at it...",
    },
    "advise": {
        "prompt": "I'm stuck between two choices and don't know what to do.",
        "completion": "That feeling of being stuck is more common than you think, and it usually means both options have real value. Let's break this down together. First, what does your gut tell you when you imagine waking up tomorrow having made each choice? Second, which option aligns more with your long-term values rather than short-term comfort? You don't need the perfect answer — you just need the next right step.",
    },
    "general": {
        "prompt": "I don't know what to do with my life right now.",
        "completion": "That's a deeply honest question, and sitting with that uncertainty takes strength. You don't need to have it all figured out today. What if we focused on just the next season — the next three months? What would feel meaningful to explore or learn during that time? Often clarity comes from action, not from thinking alone. Let's start with one small experiment this week.",
    },
}


def load_fewshot():
    global FEWSHOT_DATASET
    if os.path.exists(FEWSHOT_PATH):
        with open(FEWSHOT_PATH, encoding="utf-8") as f:
            FEWSHOT_DATASET = [json.loads(line) for line in f if line.strip()]


def save_fewshot():
    with open(FEWSHOT_PATH, "w", encoding="utf-8") as f:
        for entry in FEWSHOT_DATASET:
            f.write(json.dumps(entry) + "\n")


@app.post("/api/adaptive-reinforce")
async def adaptive_reinforce(data: dict = {}):
    global FEWSHOT_DATASET, AUTO_REINFORCED_INTENTS
    rounds = data.get("rounds", 8)
    threshold = data.get("threshold", 30)
    added = []
    for intent_key, template in FEWSHOT_TEMPLATES.items():
        if intent_key in AUTO_REINFORCED_INTENTS:
            continue
        mock_intents = {"motivate": 0, "explain": 0, "advise": 0, "general": 0}
        total_weak = 0
        for i in range(rounds):
            prompt = SIMULATION_PROMPTS[i % len(SIMULATION_PROMPTS)]
            reply = generate_mock_reply(prompt, "mentor")
            score = score_reply(reply)
            if score < 50:
                total_weak += 1
        weak_pct = (total_weak / rounds) * 100
        if weak_pct >= threshold or intent_key in ("motivate", "explain", "advise"):
            pass
        else:
            continue
        entry = {
            "intent": intent_key,
            "prompt": template["prompt"],
            "completion": template["completion"],
            "generated_at": datetime.now().isoformat(),
            "weak_rate": round(weak_pct, 1),
        }
        FEWSHOT_DATASET.append(entry)
        AUTO_REINFORCED_INTENTS.add(intent_key)
        ADAPTIVE_HISTORY.append(entry)
        added.append(entry)
    save_fewshot()
    messages = []
    if added:
        for a in added:
            messages.append(f"Auto-reinforced '{a['intent']}' (weak rate {a['weak_rate']}%).")
    else:
        messages.append("All intents already reinforced.")
    return {
        "reinforced": len(added),
        "total_fewshot": len(FEWSHOT_DATASET),
        "messages": messages,
        "added": [{"intent": a["intent"], "weak_rate": a["weak_rate"]} for a in added],
    }


@app.get("/api/adaptive-status")
async def adaptive_status():
    intents_covered = set(e["intent"] for e in FEWSHOT_DATASET)
    covered_list = sorted(intents_covered) if intents_covered else ["none"]
    return {
        "total_fewshot": len(FEWSHOT_DATASET),
        "intents_covered": covered_list,
        "auto_reinforced_intents": sorted(AUTO_REINFORCED_INTENTS),
        "history": ADAPTIVE_HISTORY[-10:],
    }


load_fewshot()

ESCALATION_REINFORCED: set = set()
ESCALATION_TEMPLATES = {
    "motivate": {"prompt": "I keep failing and I'm losing motivation completely.", "completion": "Failure is not the opposite of success — it's part of it. Let's look at what you've learned from this attempt, not just what went wrong. What's one small thing you could try differently next time? You're building resilience with every setback."},
    "explain": {"prompt": "I've read this three times and I still don't understand it at all.", "completion": "Let's try a completely different approach. Forget what you've read — let me show you with a real-world example you already understand. Think about something you know well, and I'll map the new concept onto it. Sometimes the frame of reference matters more than the explanation itself."},
    "advise": {"prompt": "Every option I consider seems terrible. I'm completely stuck.", "completion": "When every path feels wrong, it's usually because you're putting too much pressure on yourself to find the 'right' answer. Let me suggest something counterintuitive: pick the option that feels the least harmful, not the best. Sometimes unblocking yourself matters more than optimizing. You can course-correct later."},
    "general": {"prompt": "Nothing is working and I don't know what to do anymore.", "completion": "I hear how heavy this feels. Let me suggest something small but concrete: stop trying to solve everything at once. Pick one tiny thing you can control right now — literally one small action — and do only that. Progress is rebuilt one small win at a time, and you've already taken the first step by reaching out."},
}


@app.post("/api/escalation-reinforce")
async def escalation_reinforce(data: dict = {}):
    global ESCALATION_REINFORCED
    if not ESCALATION_EVENTS:
        return {"reinforced": 0, "message": "No escalation events to analyze."}
    pattern_counts = Counter((e["intent"], e["tier"]) for e in ESCALATION_EVENTS)
    added = []
    for (intent, tier), count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        key = f"{intent}_tier{tier}"
        if key in ESCALATION_REINFORCED:
            continue
        if count < 2:
            continue
        template = ESCALATION_TEMPLATES.get(intent, ESCALATION_TEMPLATES["general"])
        entry = {
            "intent": intent,
            "tier": tier,
            "escalation_count": count,
            "pattern": key,
            "prompt": template["prompt"],
            "completion": template["completion"],
            "generated_at": datetime.now().isoformat(),
        }
        FEWSHOT_DATASET.append(entry)
        ESCALATION_REINFORCED.add(key)
        ADAPTIVE_HISTORY.append({**entry, "type": "escalation_reinforce"})
        added.append(entry)
    save_fewshot()
    if added:
        messages = [f"Escalation-reinforced '{a['intent']}' tier {a['tier']} ({a['escalation_count']} events)." for a in added]
    else:
        messages = ["No escalation patterns to reinforce — all covered or below threshold."]
    return {"reinforced": len(added), "total_fewshot": len(FEWSHOT_DATASET), "messages": messages, "added": [{"intent": a["intent"], "tier": a["tier"], "count": a["escalation_count"]} for a in added]}


@app.get("/api/escalation-status")
async def escalation_status():
    patterns = Counter((e["intent"], e["tier"]) for e in ESCALATION_EVENTS)
    return {
        "total_events": len(ESCALATION_EVENTS),
        "patterns": [{"intent": k[0], "tier": k[1], "count": v} for k, v in patterns.most_common()],
        "reinforced_patterns": sorted(ESCALATION_REINFORCED),
        "reinforced_count": len(ESCALATION_REINFORCED),
    }


@app.get("/api/conflict-stats")
async def conflict_stats():
    total = len(CONFLICT_RESOLUTIONS)
    methods = Counter(r["method"] for r in CONFLICT_RESOLUTIONS)
    by_user = defaultdict(int)
    for r in CONFLICT_RESOLUTIONS:
        uid = r.get("user_id", "anonymous")
        by_user[uid] += 1
    return {
        "total": total,
        "auto": methods.get("auto", 0),
        "manual": methods.get("manual", 0),
        "gui": methods.get("gui", 0),
        "voice": methods.get("voice", 0),
        "breakdown": dict(methods.most_common()) if total else {"none": 0},
        "by_user": dict(by_user),
    }


SIMULATION_PROMPTS = [
    "I'm feeling a bit lost today. Can you help me find direction?",
    "How do I stay motivated when things get hard?",
    "Explain the concept of compounding interest to me.",
    "What should I do if I'm not sure about my career path?",
    "Tell me how to build a daily workout routine.",
    "I need advice on handling stress at work.",
    "What is the best way to learn a new skill?",
    "Encourage me to start that project I've been putting off.",
    "Can you explain why the sky is blue?",
    "Should I quit my job and start my own business?",
    "How does machine learning actually work?",
    "Give me a push to wake up early every day.",
    "I'm overwhelmed with choices. What should I focus on?",
    "Help me understand the difference between stocks and bonds.",
    "Motivate me to study for my exams.",
    "What would you recommend for someone feeling lonely?",
    "Describe how photosynthesis works in simple terms.",
    "I need a plan to get out of debt.",
    "Inspire me to write that book I've been dreaming about.",
    "Should I invest in real estate or the stock market?",
]

DEFAULT_SIMULATION_BLENDS = [
    {"name": "Mentor only", "persona": "mentor", "weights": {"mentor": 100, "coach": 0, "teacher": 0}},
    {"name": "Coach only", "persona": "coach", "weights": {"mentor": 0, "coach": 100, "teacher": 0}},
    {"name": "Teacher only", "persona": "teacher", "weights": {"mentor": 0, "coach": 0, "teacher": 100}},
    {"name": "Balanced", "persona": "mentor:34+coach:33+teacher:33", "weights": {"mentor": 34, "coach": 33, "teacher": 33}},
    {"name": "Mentor-heavy", "persona": "mentor:60+coach:20+teacher:20", "weights": {"mentor": 60, "coach": 20, "teacher": 20}},
]


@app.post("/api/simulate")
async def run_simulation(data: dict):
    rounds = data.get("rounds", 10)
    blends = data.get("blends") or DEFAULT_SIMULATION_BLENDS
    results = []
    for blend in blends:
        scores = []
        escalations = 0
        weaks = 0
        intents = Counter()
        rounds_data = []
        dy_weights = dict(blend.get("weights", {}))
        for i in range(rounds):
            prompt = SIMULATION_PROMPTS[i % len(SIMULATION_PROMPTS)]
            reply = generate_mock_reply(prompt, blend["persona"])
            score = score_reply(reply)
            intent = detect_intent(prompt)
            scores.append(score)
            if intent:
                intents[intent] += 1
            escalated = False
            if score < 50:
                weaks += 1
            if score < 50 and len(scores) >= 3 and all(s < 50 for s in scores[-3:]):
                escalations += 1
                escalated = True
            rounds_data.append({"round": i + 1, "score": score, "escalated": escalated, "intent": intent or "none"})
        results.append({
            "name": blend["name"],
            "avg_confidence": round(sum(scores) / len(scores), 1) if scores else 0,
            "escalations": escalations,
            "weak_ratio": f"{round(weaks / rounds * 100, 1)}%",
            "weak_count": weaks,
            "best_intent": intents.most_common(1)[0][0] if intents else "none",
            "persona": blend["persona"],
            "rounds_data": rounds_data,
        })
    return {"results": results, "rounds": rounds, "total_prompts": rounds * len(blends)}


@app.post("/api/federated-simulate")
async def federated_simulate(data: dict):
    rounds = data.get("rounds", 8)
    users = sorted(set(
        list(CONTRIBUTORS.keys()) +
        list(USER_WEAK_REPLIES.keys()) +
        list(USER_GOLD_EXPORTS.keys())
    ))
    if not users:
        return {"results": [], "rounds": rounds, "message": "No contributors to simulate."}
    user_results = []
    for uid in users:
        c = CONTRIBUTORS.get(uid, {})
        blends = DEFAULT_SIMULATION_BLENDS
        if c.get("sync_count", 0) > 0:
            blends = DEFAULT_SIMULATION_BLENDS[:3]
        scores = []
        total_escs = 0
        total_weaks = 0
        intents = Counter()
        for blend in blends:
            dy = dict(blend.get("weights", {}))
            for i in range(rounds):
                prompt = SIMULATION_PROMPTS[i % len(SIMULATION_PROMPTS)]
                reply = generate_mock_reply(prompt, blend["persona"])
                score = score_reply(reply)
                scores.append(score)
                intent = detect_intent(prompt)
                if intent:
                    intents[intent] += 1
                if score < 50:
                    total_weaks += 1
                if score < 50 and len(scores) >= 3 and all(s < 50 for s in scores[-3:]):
                    total_escs += 1
        total_prompts = rounds * len(blends)
        user_results.append({
            "user_id": uid,
            "sync_count": c.get("sync_count", 0),
            "conflict_count": c.get("conflict_count", 0),
            "weak_replies": USER_WEAK_REPLIES.get(uid, 0),
            "gold_exports": USER_GOLD_EXPORTS.get(uid, 0),
            "sim_avg_confidence": round(sum(scores) / len(scores), 1) if scores else 0,
            "sim_escalations": total_escs,
            "sim_weak_ratio": f"{round(total_weaks / total_prompts * 100, 1)}%" if total_prompts else "0%",
            "sim_weak_count": total_weaks,
            "sim_best_intent": intents.most_common(1)[0][0] if intents else "none",
        })
    return {"results": user_results, "rounds": rounds, "total_users": len(users)}


@app.post("/api/gold-export")
async def record_gold_export(data: dict, req: Request):
    user_id = req.headers.get("X-User-Id") or "anonymous"
    count = data.get("count", 1)
    USER_GOLD_EXPORTS[user_id] += count
    record = {"user_id": user_id, "count": count, "timestamp": datetime.now().isoformat()}
    USER_ANALYTICS_HISTORY[user_id].append(record)
    if user_id in CONTRIBUTORS:
        CONTRIBUTORS[user_id]["gold_exports"] = USER_GOLD_EXPORTS[user_id]
    return {"status": "ok", "user_id": user_id, "total_gold": USER_GOLD_EXPORTS[user_id]}


@app.get("/api/contributors")
async def get_contributors():
    enriched = []
    for c in CONTRIBUTORS.values():
        uid = c["user_id"]
        enriched.append({
            **c,
            "weak_replies": USER_WEAK_REPLIES.get(uid, 0),
            "gold_exports": USER_GOLD_EXPORTS.get(uid, 0),
        })
    return {
        "contributors": enriched,
        "total": len(enriched),
    }


@app.get("/api/contributor-analytics")
async def contributor_analytics():
    users = sorted(set(
        list(CONTRIBUTORS.keys()) +
        list(USER_WEAK_REPLIES.keys()) +
        list(USER_GOLD_EXPORTS.keys())
    ))
    analytics = []
    for uid in users:
        analytics.append({
            "user_id": uid,
            "sync_count": CONTRIBUTORS.get(uid, {}).get("sync_count", 0),
            "conflict_count": CONTRIBUTORS.get(uid, {}).get("conflict_count", 0),
            "weak_replies": USER_WEAK_REPLIES.get(uid, 0),
            "gold_exports": USER_GOLD_EXPORTS.get(uid, 0),
            "last_sync": CONTRIBUTORS.get(uid, {}).get("last_sync"),
            "history": USER_ANALYTICS_HISTORY.get(uid, [])[-5:],
        })
    return {"analytics": analytics, "total_users": len(analytics)}


@app.get("/api/contributor-narration")
async def contributor_narration():
    users = sorted(set(
        list(CONTRIBUTORS.keys()) +
        list(USER_WEAK_REPLIES.keys()) +
        list(USER_GOLD_EXPORTS.keys())
    ))
    if not users:
        return {"narration": "No contributor data recorded yet."}
    parts = []
    for uid in users:
        syncs = CONTRIBUTORS.get(uid, {}).get("sync_count", 0)
        conflicts = CONTRIBUTORS.get(uid, {}).get("conflict_count", 0)
        weaks = USER_WEAK_REPLIES.get(uid, 0)
        golds = USER_GOLD_EXPORTS.get(uid, 0)
        st = f"{uid}: {syncs} syncs"
        if conflicts: st += f", {conflicts} conflicts resolved"
        if weaks: st += f", {weaks} weak replies flagged"
        if golds: st += f", {golds} gold exports"
        parts.append(st + ".")
    narration = "Contributor analytics. " + " ".join(parts)
    return {"narration": narration, "contributors": len(users)}


@app.post("/api/conflict-resolve")
async def record_conflict_resolution(data: dict, req: Request):
    user_id = req.headers.get("X-User-Id") or "anonymous"
    method = data.get("method", "manual")
    record = {"method": method, "timestamp": datetime.now().isoformat(), "user_id": user_id}
    CONFLICT_RESOLUTIONS.append(record)
    USER_CONFLICTS[user_id].append(record)
    if user_id in CONTRIBUTORS:
        CONTRIBUTORS[user_id]["conflict_count"] = len(USER_CONFLICTS[user_id])
    else:
        CONTRIBUTORS[user_id] = {
            "user_id": user_id,
            "last_sync": None,
            "sync_count": 0,
            "conflict_count": len(USER_CONFLICTS[user_id]),
            "status": "active",
        }
    return {"status": "ok", "method": method, "user_id": user_id}


@app.get("/{path:path}")
async def serve_frontend(path: str):
    if path.startswith("api/"):
        raise HTTPException(404)
    idx = os.path.join(HERE, "index.html")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>youfeel</h1><p>Frontend not found.</p>")
