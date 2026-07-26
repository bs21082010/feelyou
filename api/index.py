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
from fastapi.responses import FileResponse, HTMLResponse, Response
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
COVERAGE_HISTORY: list = []
INTENT_ESCALATION_COUNTER: dict = defaultdict(int)
PREV_FEWSHOT_COUNT = 0
PREV_CONFLICT_COUNT = 0
PREV_GOLD_COUNT = 0
PREV_LEADERBOARD = None
PREV_BADGE_MAP = {}
LATEST_FED_SIM_RESULTS = None
PERSONA_BALANCE = {"mentor": 50, "coach": 25, "teacher": 25}
AUTO_BALANCE_ENABLED = True
PRE_BALANCE_CONFIDENCE = None
FEWSHOT_PATH = os.path.join(HERE, "fewshot_dataset.jsonl")
USER_WEAK_REPLIES: dict = defaultdict(int)
USER_GOLD_EXPORTS: dict = defaultdict(int)
USER_ANALYTICS_HISTORY: dict = defaultdict(list)
PERSONA_DRIFT_LOG: list = []
HYBRID_MODE = "local"
HYBRID_OFFLINE = False

DRIFT_THRESHOLD = 15
SCORE_THRESHOLD = 70
WEIGHT_BOOST_FACTOR = 12
WEIGHT_DECAY_PER_TURN = 3
PREDICTIVE_BOOST = 8
CONFLICT_AUTO_POLICY = "low"
DYNAMIC_BOOST = {"persona": "mentor", "reason": "default"}
BADGE_HISTORY: list = []
PREV_BOOST = None
WEIGHT_HISTORY: list = []

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
    emotion: Optional[str] = None
    dynamic_boost: Optional[dict] = None


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


def record_persona_drift(user_id: str, action_type: str, weights: dict = None, sim_avg: float = None):
    now = datetime.now().isoformat()
    snap = dict(weights) if weights else {}
    PERSONA_DRIFT_LOG.append({
        "timestamp": now,
        "user_id": user_id,
        "action": action_type,
        "weights": snap,
        "sim_avg": sim_avg,
    })


def compute_narration_emotion(avg_conf=None, escalations=None, consecutive_low=None, coverage_trend=None):
    if escalations and escalations >= 5: return "empathy"
    if consecutive_low and consecutive_low >= 3: return "challenge"
    if avg_conf and avg_conf >= 75: return "joy"
    if coverage_trend and coverage_trend > 0: return "encouragement"
    return "reflection"


async def auto_apply_balance():
    global PERSONA_BALANCE, PRE_BALANCE_CONFIDENCE, AUTO_BALANCE_ENABLED
    if not AUTO_BALANCE_ENABLED:
        return {"applied": False, "reason": "auto-balance disabled"}
    tune_resp = await adaptive_tuning()
    suggested = tune_resp.get("suggested_adjustments", {})
    if not suggested:
        return {"applied": False, "reason": "no adjustments needed"}
    PRE_BALANCE_CONFIDENCE = round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None
    for persona, adj in suggested.items():
        PERSONA_BALANCE[persona] = min(100, max(0, PERSONA_BALANCE.get(persona, 50) + adj))
    normalize_weights(PERSONA_BALANCE)
    return {
        "applied": True,
        "adjustments": suggested,
        "balance": dict(PERSONA_BALANCE),
        "pre_balance_confidence": PRE_BALANCE_CONFIDENCE,
    }


async def check_rollback():
    global PERSONA_BALANCE, PRE_BALANCE_CONFIDENCE
    if PRE_BALANCE_CONFIDENCE is None:
        return {"rolled_back": False}
    current_avg = round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None
    if current_avg is not None and current_avg < PRE_BALANCE_CONFIDENCE - 5:
        PERSONA_BALANCE = {"mentor": 50, "coach": 25, "teacher": 25}
        PRE_BALANCE_CONFIDENCE = None
        return {"rolled_back": True, "reason": f"Confidence dropped from {PRE_BALANCE_CONFIDENCE} to {current_avg}"}
    return {"rolled_back": False}


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


EMOTION_REPLIES = {
    "joy": [
        "That's wonderful! I can feel the positivity in your words. Moments like these are what make the journey worthwhile. Hold onto this feeling and let it fuel your next step.",
        "I love this energy! There's something special about when things click into place. You've earned this moment — enjoy it fully.",
        "This makes me genuinely happy to hear. You're exactly where you need to be, and the best part is you're recognizing it. That's a superpower.",
        "Yes! That's the spirit. Celebrating wins, big or small, rewires your brain for more success. Savor this moment.",
        "Love this! You're lighting up and it's contagious. Keep that momentum going — you're building something real.",
        "This is the kind of news that makes everything worthwhile. You're not just moving forward, you're thriving. Soak it in.",
        "That's amazing! The energy you're putting out is coming back to you. Keep riding this wave.",
        "I'm genuinely excited for you. These breakthroughs aren't accidents — they're the result of your consistency. Well done.",
        "What a beautiful perspective. When you approach life this way, everything becomes an opportunity. You're doing it right.",
        "This warms my heart. You're not just succeeding — you're growing in ways that will serve you for years. Be proud.",
    ],
    "empathy": [
        "I hear you, and I want you to know that what you're feeling is completely valid. You don't have to have it all figured out right now. Just taking the time to express it is a powerful step.",
        "That sounds really hard, and I'm sorry you're going through it. Please remember: you don't have to carry this alone. Even just saying it out loud makes the load a little lighter.",
        "I can feel the weight of what you're sharing, and I want you to know I'm here with you. You're not broken for feeling this way — you're human. And humans get through things like this together.",
        "That must be incredibly difficult. I see you, I hear you, and I'm not going anywhere. Let's sit with this feeling for a moment — sometimes having someone witness our pain is the first step toward healing.",
        "I wish I could give you a hug right now. What you're describing is real, and it matters. You matter. Let's take this one breath at a time.",
        "You're carrying so much, and yet here you are, still showing up. That takes more strength than you realize. I see you.",
        "It's okay to not be okay. Seriously. You don't need to perform strength right now. Just be exactly where you are, and I'll meet you there.",
        "I appreciate you trusting me with this. Sharing vulnerability isn't weakness — it's one of the bravest things you can do. I'm honored you'd share it with me.",
        "What you're feeling makes total sense given what you've been through. Anyone in your shoes would feel the same way. Be gentle with yourself right now.",
        "I'm here, I'm listening, and I care. You don't need to filter or minimize what you're feeling. Let it out. That's what I'm here for.",
    ],
    "encouragement": [
        "You've come so much further than you give yourself credit for. Look back at where you started — the growth is real. Keep going. You're closer than you think.",
        "I believe in you. And not in some generic way — I genuinely believe you have what it takes. The doubt you feel is just proof that you're pushing past your comfort zone. That's where growth happens.",
        "Progress isn't always visible in the moment, but trust me — every step counts. You're building something meaningful, and future you is going to be so grateful you kept going.",
        "You are capable of more than you know. The only difference between where you are and where you want to be is one more attempt. Just one more.",
        "This is the hard part, and you're doing it anyway. That's not luck — that's character. Keep pushing. The other side is worth it.",
        "I've seen you navigate hard things before and come through. This is no different. You have a track record of resilience, even if you don't see it.",
        "You don't need to be perfect. You just need to be persistent. And from what I can see, you've got that in spades. Keep showing up.",
        "The fact that you're still trying tells me everything I need to know about your strength. Most people would have quit by now. Not you. That matters.",
        "Let me remind you of something: every expert was once a beginner. Every success story has chapters of struggle. You're writing yours right now.",
        "You've got more strength than you realize. Sometimes we forget our own resilience because we're so focused on what's ahead. But look at what you've already overcome.",
    ],
    "challenge": [
        "Okay, let's level up. You've been playing it safe — time to take a real risk. What's one thing you've been avoiding that you know you need to do?",
        "I'm going to push you a little here: you already know the answer. You've known it for a while. What's stopping you isn't lack of information — it's fear of the unknown. Let's address that head-on.",
        "Here's a hard truth: comfort zones are where dreams go to die. You didn't come this far to play small. What's the boldest version of your next step?",
        "Stop waiting for the perfect moment. It doesn't exist. The best time to start was yesterday, the second best time is right now. What's one thing you can do in the next five minutes?",
        "I'm not going to tell you what you want to hear — I'm going to tell you what you need to hear. You're underselling yourself. Raise your standards.",
        "You asked for my honest take, so here it is: you're overthinking this. The answer is simpler than you're making it. Trust your gut and act.",
        "Let's get real for a second. The version of you that achieves this goal doesn't make excuses. That version just finds a way. Are you ready to become that person?",
        "I see potential in you, and that's exactly why I'm pushing. Potential without action is just a wish. Let's turn it into a plan.",
        "Here's the uncomfortable truth: if it doesn't challenge you, it doesn't change you. This is your opportunity to grow. Don't waste it playing small.",
        "You asked for my honest opinion, so here it is: you can do this. But not with your current approach. Something has to change. Are you ready for that?",
    ],
    "reflection": [
        "That's worth sitting with. Not every question needs an immediate answer. Sometimes the most powerful thing is to let the question resonate and see what emerges.",
        "I love this kind of depth. You're not just looking for surface answers — you're exploring. That curiosity is one of your greatest strengths.",
        "Let's pause and reflect on that. What does this situation reveal about what you truly value? Often our reactions tell us more about ourselves than about the circumstances.",
        "That's a profound observation. The fact that you're thinking about this at this level tells me you're ready for deeper understanding. Let's explore it together.",
        "Sometimes the best response isn't an answer — it's a better question. What if you looked at this from the opposite perspective? What would change?",
        "You've given me a lot to think about too. This kind of conversation is rare and valuable. Thank you for bringing this depth.",
        "Let's zoom out for a moment. How will this feel a year from now? Sometimes distance gives us clarity that immediacy obscures.",
        "That's a really thoughtful way to put it. I think what you're really asking is about meaning, not just mechanics. Let's explore the 'why' behind the 'what.'",
        "I appreciate the depth of this question. It shows you're not just looking for quick answers — you want real understanding. That's rare and valuable.",
        "This reminds me of something important: growth often happens in the quiet moments of reflection, not in constant action. Thank you for creating space for that here.",
    ],
    "neutral": [
        "That's an interesting point. Let me think about that with you. What aspect would you like to explore first?",
        "I appreciate you sharing that. There's a lot to unpack here — where would you like to start?",
        "That's a good question. Let me share a thought and you tell me if it resonates with your experience.",
        "I'm glad you brought that up. It's one of those topics where the context really matters. Can you tell me more about your specific situation?",
        "That's worth exploring. I have a few angles we could look at — which one interests you most?",
        "I hear you. Let me reflect back what I'm understanding to make sure I'm on the right track.",
        "Great point. This connects to a broader pattern I've noticed. Would you like me to share that perspective?",
        "Thanks for saying that. It gives me a clearer picture of where you're coming from. Let's build on that.",
        "That's an interesting angle. I hadn't thought about it that way. What else comes to mind when you consider this?",
        "I appreciate the clarity. This gives us a solid foundation to work from. What's the next layer you want to explore?",
    ],
}

EMOTION_KEYWORDS = {
    "joy": ["happy", "great", "amazing", "wonderful", "love", "excited", "fantastic", "awesome", "thrilled", "blessed", "grateful", "proud", "celebrate", "beautiful", "best"],
    "empathy": ["sad", "struggl", "hard", "difficult", "tired", "hurt", "lonely", "overwhelm", "depress", "anxious", "worried", "scared", "pain", "heavy", "exhaust"],
    "encouragement": ["try", "hope", "believe", "keep", "continue", "persist", "determin", "faith", "possible", "dream", "goal", "aspir", "motivat", "inspire", "push"],
    "challenge": ["boring", "stuck", "complacent", "coast", "plateau", "lazy", "fear", "avoid", "procrastinat", "comfort zone", "settl", "waste", "risk", "bold", "dare"],
    "reflection": ["why", "mean", "purpose", "meaning", "wonder", "philosoph", "think", "consider", "reflect", "perspective", "lesson", "deeper", "question", "curious", "understand"],
}


def detect_emotion(text: str) -> str:
    lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        scores[emotion] = sum(1 for kw in keywords if kw in lower)
    if any(scores.values()):
        return max(scores, key=scores.get)
    return "neutral"


def generate_mock_reply(user_message: str, persona_name: str):
    h = sum(ord(c) * (i + 1) for i, c in enumerate(user_message))
    emotion = detect_emotion(user_message)
    intent = detect_intent(user_message)
    pool = EMOTION_REPLIES.get(emotion, EMOTION_REPLIES["neutral"])
    base = pool[h % len(pool)]
    if intent == "motivate" and emotion != "joy":
        base += " You've got the strength to move through this — I believe in you."
    elif intent == "explain":
        base += " Let me know if you'd like me to break any part down further."
    elif intent == "advise" and emotion in ("empathy", "neutral"):
        base += " Would it help to talk through some options together?"
    if "coach" in persona_name:
        base += " Now, what's your next step going to be?"
    elif "teacher" in persona_name:
        base += " Does that help clarify things?"
    else:
        base += " How does that land with you?"
    return base, emotion


def call_llm(system_prompt: str, user_message: str, persona_name: str = "mentor"):
    global HYBRID_OFFLINE
    headers = {"Content-Type": "application/json"}
    cloud_url = os.environ.get("LLM_CLOUD_URL", "https://api.openai.com/v1/chat/completions")
    cloud_model = os.environ.get("LLM_CLOUD_MODEL", "gpt-4o-mini")
    target_url = cloud_url if HYBRID_MODE == "cloud" else LLM_API_URL
    target_model = cloud_model if HYBRID_MODE == "cloud" else LLM_MODEL
    api_key = os.environ.get("LLM_CLOUD_KEY") if HYBRID_MODE == "cloud" else LLM_API_KEY
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps({
        "model": target_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }).encode()
    req = urllib.request.Request(target_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            reply = data.get("message", {}).get("content", "") or data.get("choices", [{}])[0].get("message", {}).get("content", "") or FALLBACK
            return reply, None
    except Exception as e:
        if HYBRID_MODE == "cloud":
            HYBRID_OFFLINE = True
        mock, emotion = generate_mock_reply(user_message, persona_name)
        return mock, emotion


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    global RECENT_SCORES, CONSECUTIVE_LOW, LAST_REPLY, LAST_PROMPT, LAST_SCORE, WEAK_REPLY_COUNT, ESCALATION_EVENTS
    user_id = request.headers.get("X-User-Id") or "anonymous"

    weights = req.weights or dict(PERSONA_BALANCE)
    prompt = apply_template(req.message, weights)
    persona = build_persona(req.persona, weights)

    reply, emotion = call_llm(persona, prompt, req.persona)
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
        drift_avg = round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None
        record_persona_drift(user_id, "weak_reply", weights, drift_avg)
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
        INTENT_ESCALATION_COUNTER[intent_key] += 1

    avg_conf = round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else 100
    global DYNAMIC_BOOST, PREV_BOOST
    mode_tag = "cloud" if HYBRID_MODE == "cloud" and not HYBRID_OFFLINE else "local"
    if CONSECUTIVE_LOW >= 5:
        boost_amt = 10 if mode_tag == "cloud" else 8
        DYNAMIC_BOOST = {"persona": "coach", "reason": f"consecutive low scores ({CONSECUTIVE_LOW})", "boost": boost_amt, "source": mode_tag}
    elif CONSECUTIVE_LOW >= 3:
        boost_amt = 6 if mode_tag == "cloud" else 8
        DYNAMIC_BOOST = {"persona": "mentor", "reason": "escalation detected", "boost": boost_amt, "source": mode_tag}
    elif avg_conf >= 80 and intent == "explain":
        boost_amt = 10 if mode_tag == "cloud" else 8
        DYNAMIC_BOOST = {"persona": "teacher", "reason": "high confidence + explanation intent", "boost": boost_amt, "source": mode_tag}
    elif avg_conf >= 80:
        boost_amt = 4 if mode_tag == "cloud" else 6
        DYNAMIC_BOOST = {"persona": "mentor", "reason": "high confidence baseline", "boost": boost_amt, "source": mode_tag}
    else:
        DYNAMIC_BOOST = {"persona": "mentor", "reason": "default", "boost": 0, "source": mode_tag}
    boost = DYNAMIC_BOOST["persona"]
    boost_amt = DYNAMIC_BOOST.get("boost", 8)
    if boost in weights and boost_amt:
        weights[boost] = min(100, weights.get(boost, 50) + boost_amt)
    if DYNAMIC_BOOST != PREV_BOOST and PREV_BOOST is not None and DYNAMIC_BOOST.get("reason") != "default":
        DYNAMIC_BOOST["changed"] = True
    else:
        DYNAMIC_BOOST["changed"] = False
    global PREV_BOOST
    PREV_BOOST = dict(DYNAMIC_BOOST)

    if weights:
        normalize_weights(weights)
    WEIGHT_HISTORY.append({"timestamp": datetime.now().isoformat(), "weights": dict(weights) if weights else {}, "boost": dict(DYNAMIC_BOOST)})

    return ChatResponse(
        reply=reply,
        score=score,
        weights=weights,
        intent=intent,
        escalation_tier=escalation_tier,
        emotion=emotion,
        dynamic_boost=dict(DYNAMIC_BOOST),
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
    global LAST_SYNC_TIME, LAST_SYNC_STATUS, SYNC_COUNT, PREV_FEWSHOT_COUNT, PREV_CONFLICT_COUNT, PREV_GOLD_COUNT
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
    avg_c = round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None
    record_persona_drift(user_id, "sync", sim_avg=avg_c)
    new_fewshot = len(FEWSHOT_DATASET) - PREV_FEWSHOT_COUNT
    new_conflicts = len(CONFLICT_RESOLUTIONS) - PREV_CONFLICT_COUNT
    new_gold = sum(USER_GOLD_EXPORTS.values()) - PREV_GOLD_COUNT
    PREV_FEWSHOT_COUNT = len(FEWSHOT_DATASET)
    PREV_CONFLICT_COUNT = len(CONFLICT_RESOLUTIONS)
    PREV_GOLD_COUNT = sum(USER_GOLD_EXPORTS.values())
    bal = await auto_apply_balance()
    roll = await check_rollback()
    return {
        "status": "ok",
        "synced_at": LAST_SYNC_TIME,
        "sync_count": SYNC_COUNT,
        "total_scores": len(RECENT_SCORES),
        "weak_count": WEAK_REPLY_COUNT,
        "avg_confidence": avg_c,
        "user_id": user_id,
        "contributor": CONTRIBUTORS[user_id],
        "dataset_changes": {
            "fewshot_entries_added": max(0, new_fewshot),
            "conflicts_resolved": max(0, new_conflicts),
            "gold_exports": max(0, new_gold),
        },
        "balance_applied": bal.get("applied", False),
        "balance_adjustments": bal.get("adjustments"),
        "rollback_applied": roll.get("rolled_back", False),
        "rollback_reason": roll.get("reason"),
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
            reply, _ = generate_mock_reply(prompt, "mentor")
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
    COVERAGE_HISTORY.append({
        "timestamp": datetime.now().isoformat(),
        "total_fewshot": len(FEWSHOT_DATASET),
        "escalation_count": sum(INTENT_ESCALATION_COUNTER.values()),
        "weak_intent_count": len([e for e in TRAINING_EVENTS if e["type"] == "weak_reply"]),
        "auto_reinforced": sorted(AUTO_REINFORCED_INTENTS),
        "escalation_reinforced": sorted(ESCALATION_REINFORCED),
        "intents_covered": sorted(set(e["intent"] for e in FEWSHOT_DATASET)),
    })
    save_fewshot()
    messages = []
    if added:
        for a in added:
            messages.append(f"Auto-reinforced '{a['intent']}' (weak rate {a['weak_rate']}%).")
    else:
        messages.append("All intents already reinforced.")
    bal = await auto_apply_balance()
    roll = await check_rollback()
    return {
        "reinforced": len(added),
        "total_fewshot": len(FEWSHOT_DATASET),
        "messages": messages,
        "added": [{"intent": a["intent"], "weak_rate": a["weak_rate"]} for a in added],
        "balance_applied": bal.get("applied", False),
        "balance_adjustments": bal.get("adjustments"),
        "rollback_applied": roll.get("rolled_back", False),
        "rollback_reason": roll.get("reason"),
    }


@app.get("/api/adaptive-status")
async def adaptive_status():
    intents_covered = set(e["intent"] for e in FEWSHOT_DATASET)
    covered_list = sorted(intents_covered) if intents_covered else ["none"]
    return {
        "total_fewshot": len(FEWSHOT_DATASET),
        "intents_covered": covered_list,
        "auto_reinforced_intents": sorted(AUTO_REINFORCED_INTENTS),
        "escalation_intent_counts": dict(sorted(INTENT_ESCALATION_COUNTER.items(), key=lambda x: -x[1])),
        "history": ADAPTIVE_HISTORY[-10:],
        "coverage_history": COVERAGE_HISTORY[-30:],
    }


load_fewshot()

@app.get("/api/coverage-history")
async def coverage_history():
    return {
        "coverage_history": COVERAGE_HISTORY[-50:],
        "total_fewshot": len(FEWSHOT_DATASET),
        "total_escalations": sum(INTENT_ESCALATION_COUNTER.values()),
        "total_weak": len([e for e in TRAINING_EVENTS if e["type"] == "weak_reply"]),
    }


@app.get("/api/drift-narration")
async def drift_narration():
    if not PERSONA_DRIFT_LOG:
        return {"narration": "No persona drift data recorded yet."}
    by_user = defaultdict(list)
    for entry in PERSONA_DRIFT_LOG:
        by_user[entry["user_id"]].append(entry)
    parts = []
    for uid, logs in sorted(by_user.items()):
        if len(logs) < 2 and not any(l["weights"] for l in logs):
            continue
        weights_before = {}
        weights_after = {}
        for lg in logs:
            if lg["weights"]:
                weights_after = dict(lg["weights"])
                if not weights_before:
                    weights_before = dict(lg["weights"])
        deltas = {}
        all_keys = set(weights_before.keys()) | set(weights_after.keys())
        for k in sorted(all_keys):
            before = weights_before.get(k, 50)
            after = weights_after.get(k, 50)
            diff = round(after - before, 1)
            if diff != 0:
                deltas[k] = diff
        actions = Counter(lg["action"] for lg in logs)
        reasons = []
        for act, cnt in actions.most_common(3):
            reasons.append(f"{cnt} {act}")
        if deltas:
            delta_str = ", ".join(f"{k} {'drifted' if abs(v) > 0 else 'changed'} {v:+}" for k, v in deltas.items())
            parts.append(f"{uid}'s contributions caused {delta_str}, from {len(logs)} events including " + ", ".join(reasons) + ".")
        elif len(logs) > 0:
            parts.append(f"{uid} had {len(logs)} events without weight changes.")
    if not parts:
        return {"narration": "No significant persona drift detected yet."}
    return {"narration": "Persona drift analysis. " + " ".join(parts), "events": len(PERSONA_DRIFT_LOG)}


@app.get("/api/leaderboard")
async def leaderboard():
    users = sorted(set(
        list(CONTRIBUTORS.keys()) +
        list(USER_WEAK_REPLIES.keys()) +
        list(USER_GOLD_EXPORTS.keys())
    ))
    if not users:
        return {"leaderboard": [], "message": "No contributors yet."}
    entries = []
    for uid in users:
        c = CONTRIBUTORS.get(uid, {})
        syncs = c.get("sync_count", 0)
        conflicts = c.get("conflict_count", 0)
        weaks = USER_WEAK_REPLIES.get(uid, 0)
        golds = USER_GOLD_EXPORTS.get(uid, 0)
        drift_logs = [e for e in PERSONA_DRIFT_LOG if e["user_id"] == uid]
        esc_count = len(drift_logs)
        action_counts = Counter(lg["action"] for lg in drift_logs)
        score = syncs * 10 + golds * 25 - weaks * 5 - conflicts * 3 + esc_count * 2
        badge_list = []
        if score > 0:
            badge_list.append("\U0001F3C6" if score >= 50 else "\U0001F947" if score >= 30 else "\U0001F948" if score >= 15 else "")
        if golds >= 3:
            badge_list.append("\u2B50")
        if syncs >= 5:
            badge_list.append("\U0001F504")
        if weaks == 0 and syncs > 0:
            badge_list.append("\u2705")
        if esc_count >= 3:
            badge_list.append("\u26A1")
        if action_counts.get("sync", 0) + action_counts.get("conflict_resolve", 0) + action_counts.get("weak_reply", 0) + action_counts.get("gold_export", 0) >= 5:
            badge_list.append("\U0001F3A4")
        if action_counts.get("simulate", 0) + action_counts.get("federated_simulate", 0) > 0 or syncs >= 3:
            badge_list.append("\U0001F9E9")
        if action_counts.get("conflict_resolve", 0) >= 2:
            badge_list.append("\U0001F91D")
        if golds >= 1:
            badge_list.append("\U0001F31F")
        badge_list = [b for b in badge_list if b]
        entries.append({
            "user_id": uid,
            "score": score,
            "syncs": syncs,
            "conflicts": conflicts,
            "weak_replies": weaks,
            "gold_exports": golds,
            "events": esc_count,
            "action_counts": dict(action_counts),
            "badges": badge_list,
        })
    ranked = sorted(entries, key=lambda e: e["score"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return {"leaderboard": ranked, "total": len(ranked)}


@app.get("/api/contributor-impact")
async def contributor_impact():
    global PREV_LEADERBOARD
    lb_resp = await leaderboard()
    entries = lb_resp.get("leaderboard", [])
    if not entries:
        return {"narration": "No contributors recorded yet."}
    parts = []
    top = entries[0]
    badge_names = {"\U0001F3C6": "Trophy", "\U0001F947": "Gold", "\U0001F948": "Silver", "\u2B50": "Star", "\U0001F504": "Syncer", "\u2705": "Clean", "\u26A1": "Energizer", "\U0001F3A4": "Narrator", "\U0001F9E9": "Strategist", "\U0001F91D": "Diplomat", "\U0001F31F": "Pioneer"}
    top_badges = [badge_names.get(b, b) for b in top["badges"] if b]
    parts.append(f"{top['user_id']} is leading with {top['score']} points" + (f", earning {', '.join(top_badges)} badges" if top_badges else "") + ".")
    changes = {}
    if PREV_LEADERBOARD:
        prev_top = PREV_LEADERBOARD[0] if PREV_LEADERBOARD else None
        if prev_top and prev_top["user_id"] != top["user_id"]:
            changes["new_leader"] = f"{top['user_id']} took the lead from {prev_top['user_id']}."
            parts.append(changes["new_leader"])
        if prev_top and top["score"] > prev_top["score"]:
            diff = top["score"] - prev_top["score"]
            changes["score_gain"] = diff
            parts.append(f"That is {diff} points higher than last check.")
        prev_badges = set(prev_top.get("badges", [])) if prev_top else set()
        curr_badges = set(top.get("badges", []))
        new_badges = curr_badges - prev_badges
        if new_badges:
            new_names = [badge_names.get(b, b) for b in new_badges]
            changes["new_badges"] = list(new_badges)
            parts.append(f"They just earned {', '.join(new_names)}.")
    else:
        parts.append("This is their first appearance on the leaderboard.")
    PREV_LEADERBOARD = entries
    if len(entries) > 1:
        second = entries[1]
        parts.append(f"{second['user_id']} is second with {second['score']} points.")
    if len(entries) > 2:
        third = entries[2]
        parts.append(f"{third['user_id']} rounds out the top three with {third['score']} points.")
    total_events = sum(e["events"] for e in entries)
    total_syncs = sum(e["syncs"] for e in entries)
    if total_events:
        parts.append(f"Across {len(entries)} contributors: {total_syncs} total syncs, {total_events} drift events logged.")
    return {"narration": "Contributor impact. " + " ".join(parts), "leaderboard": entries[:5], "changes": changes}


@app.get("/api/adaptive-tuning")
async def adaptive_tuning():
    suggested = {}
    drift_mentor_deltas = []
    drift_coach_deltas = []
    drift_teacher_deltas = []
    drift_logs_with_weights = [e for e in PERSONA_DRIFT_LOG if e["weights"]]
    if drift_logs_with_weights:
        first_w = drift_logs_with_weights[0]["weights"]
        last_w = drift_logs_with_weights[-1]["weights"]
        for p in ["mentor", "coach", "teacher"]:
            delta = last_w.get(p, 50) - first_w.get(p, 50)
            if p == "mentor":
                drift_mentor_deltas.append(delta)
            elif p == "coach":
                drift_coach_deltas.append(delta)
            else:
                drift_teacher_deltas.append(delta)
    reinforced = set(e["intent"] for e in FEWSHOT_DATASET)
    if "motivate" in reinforced:
        suggested["coach"] = suggested.get("coach", 0) + 5
    if "explain" in reinforced:
        suggested["teacher"] = suggested.get("teacher", 0) + 5
    if "advise" in reinforced:
        suggested["mentor"] = suggested.get("mentor", 0) + 5
    esc_intents = dict(INTENT_ESCALATION_COUNTER)
    for intent, count in esc_intents.items():
        if count >= 2 and intent == "motivate":
            suggested["coach"] = suggested.get("coach", 0) + 3
        elif count >= 2 and intent == "explain":
            suggested["teacher"] = suggested.get("teacher", 0) + 3
        elif count >= 2 and intent == "advise":
            suggested["mentor"] = suggested.get("mentor", 0) + 3
    if drift_mentor_deltas and drift_mentor_deltas[-1] < -5:
        suggested["mentor"] = suggested.get("mentor", 0) + 8
    if drift_coach_deltas and drift_coach_deltas[-1] < -5:
        suggested["coach"] = suggested.get("coach", 0) + 8
    if drift_teacher_deltas and drift_teacher_deltas[-1] < -5:
        suggested["teacher"] = suggested.get("teacher", 0) + 8
    messages = []
    if suggested:
        for p, adj in sorted(suggested.items(), key=lambda x: -x[1]):
            messages.append(f"Boost {p} by {adj} points due to " + (
                "reinforcement patterns" if any(p == k for k in ["coach", "teacher", "mentor"] if p == k and any(
                    intent in reinforced for intent in {"motivate": "coach", "explain": "teacher", "advise": "mentor"}.get(p, set())
                )) else "drift correction"
            ))
    else:
        messages.append("No tuning needed — all personas are balanced.")
    return {
        "suggested_adjustments": suggested,
        "current_drift": {
            "mentor": drift_mentor_deltas[-1] if drift_mentor_deltas else 0,
            "coach": drift_coach_deltas[-1] if drift_coach_deltas else 0,
            "teacher": drift_teacher_deltas[-1] if drift_teacher_deltas else 0,
        },
        "messages": messages or ["No adjustments needed."],
    }


@app.post("/api/auto-balance")
async def auto_balance(data: dict = {}):
    global PERSONA_BALANCE, PRE_BALANCE_CONFIDENCE
    tune_resp = await adaptive_tuning()
    suggested = tune_resp.get("suggested_adjustments", {})
    if not suggested:
        return {"applied": False, "message": "No adjustments needed.", "balance": PERSONA_BALANCE}
    PRE_BALANCE_CONFIDENCE = round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None
    for persona, adj in suggested.items():
        PERSONA_BALANCE[persona] = min(100, max(0, PERSONA_BALANCE.get(persona, 50) + adj))
    normalize_weights(PERSONA_BALANCE)
    return {
        "applied": True,
        "message": f"Applied adjustments: {', '.join(f'{p} +{suggested[p]}' for p in suggested)}.",
        "balance": PERSONA_BALANCE,
        "pre_balance_confidence": PRE_BALANCE_CONFIDENCE,
        "adjustments": suggested,
    }


@app.post("/api/rollback-balance")
async def rollback_balance():
    global PERSONA_BALANCE, PRE_BALANCE_CONFIDENCE
    PERSONA_BALANCE = {"mentor": 50, "coach": 25, "teacher": 25}
    current_avg = round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None
    degraded = PRE_BALANCE_CONFIDENCE is not None and current_avg is not None and current_avg < PRE_BALANCE_CONFIDENCE - 5
    PRE_BALANCE_CONFIDENCE = None
    return {
        "rolled_back": True,
        "balance": PERSONA_BALANCE,
        "degraded": degraded,
        "message": "Balance rolled back to defaults." + (" Confidence dropped since balance was applied." if degraded else ""),
    }


@app.post("/api/federated-sim-narration")
async def federated_sim_narration(data: dict = {}):
    rounds = data.get("rounds", 8)
    fs_resp = await federated_simulate({"rounds": rounds})
    results = fs_resp.get("results", [])
    ranked = fs_resp.get("ranked", [])
    if not results:
        return {"narration": "No contributors to simulate."}
    parts = []
    if ranked:
        top_r = ranked[0]
        parts.append(f"{top_r['user_id']} leads with {top_r['avg_confidence']} percent average confidence and {top_r['escalations']} escalations.")
        if len(ranked) > 1:
            second_r = ranked[1]
            diff = round(top_r["avg_confidence"] - second_r["avg_confidence"], 1)
            esc_diff = top_r["escalations"] - second_r["escalations"]
            parts.append(f"That is {diff} points higher than {second_r['user_id']}")
            if esc_diff < 0:
                parts.append(f"who had {abs(esc_diff)} fewer escalations.")
            else:
                parts.append(".")
    for r in results[:3]:
        if r.get("blend_scores"):
            best_blend = max(r["blend_scores"], key=lambda b: b["avg"])
            balanced_score = next((b["avg"] for b in r["blend_scores"] if "balanced" in b["blend"].lower()), None)
            parts.append(f"{r['user_id']}'s best blend was {best_blend['blend']} at {best_blend['avg']} percent confidence.")
            if balanced_score is not None and best_blend["avg"] > balanced_score:
                boost = round(best_blend["avg"] - balanced_score, 1)
                parts.append(f"Outperforming balanced blend by {boost} points.")
    avg_conf = fs_resp.get("fed_avg")
    if avg_conf:
        parts.append(f"Federated average confidence is {avg_conf} percent across {len(results)} users.")
    emotion = compute_narration_emotion(avg_conf=avg_conf, escalations=sum(r["escalations"] for r in ranked) if ranked else 0)
    return {"narration": "Federated simulation results. " + " ".join(parts), "ranked": ranked[:5], "fed_avg": avg_conf, "emotion": emotion}


@app.get("/api/unified-story")
async def unified_story():
    chapters = []

    # Chapter 1: Drift overview
    drift_users = set(e["user_id"] for e in PERSONA_DRIFT_LOG if e["user_id"] not in ("simulation", "federated"))
    drift_count = len([e for e in PERSONA_DRIFT_LOG if e["user_id"] not in ("simulation", "federated")])
    if drift_users:
        chapters.append(f"The journey spans {len(drift_users)} contributors with {drift_count} drift events.")
        if len(PERSONA_DRIFT_LOG) >= 2 and any(e["weights"] for e in PERSONA_DRIFT_LOG[-2:]):
            recent = [e for e in PERSONA_DRIFT_LOG[-3:] if e["weights"]]
            if recent:
                last_w = recent[-1]["weights"]
                w_str = ", ".join(f"{k}: {v}" for k, v in sorted(last_w.items()))
                chapters.append(f"Latest persona weights are {w_str}.")
    else:
        chapters.append("No drift events recorded yet.")

    # Chapter 2: Leaderboard snapshot
    lb_resp = await leaderboard()
    entries = lb_resp.get("leaderboard", [])
    if entries:
        top3 = entries[:3]
        chapters.append(f"On the leaderboard, {top3[0]['user_id']} leads with {top3[0]['score']} points.")
        if len(top3) > 1:
            chapters.append(f"Followed by {top3[1]['user_id']} with {top3[1]['score']} points")
            if len(top3) > 2:
                chapters.append(f"and {top3[2]['user_id']} with {top3[2]['score']} points.")
            else:
                chapters[-1] += "."
        total_syncs = sum(e["syncs"] for e in entries)
        total_events = sum(e["events"] for e in entries)
        chapters.append(f"Collectively, they have {total_syncs} syncs and {total_events} drift events.")
    else:
        chapters.append("The leaderboard is empty.")

    # Chapter 3: Contributor highlights
    if entries:
        for e in entries[:3]:
            parts_h = []
            drift_actions = [d for d in PERSONA_DRIFT_LOG if d["user_id"] == e["user_id"] and d["action"] != "federated_simulate"]
            p_deltas = {}
            for d in drift_actions:
                if d.get("weights"):
                    for p, v in d["weights"].items():
                        p_deltas[p] = v
            if p_deltas:
                delta_str = ", ".join(f"{p} {v}" for p, v in sorted(p_deltas.items()))
                parts_h.append(f"{e['user_id']} shaped weights to {delta_str}")
            if e.get("badges"):
                badge_names = {"\U0001F3C6": "Trophy", "\U0001F947": "Gold", "\U0001F948": "Silver", "\u2B50": "Star", "\U0001F504": "Syncer", "\u2705": "Clean", "\u26A1": "Energizer", "\U0001F3A4": "Narrator", "\U0001F9E9": "Strategist", "\U0001F91D": "Diplomat", "\U0001F31F": "Pioneer"}
                names = [badge_names.get(b, b) for b in e["badges"] if b]
                if names:
                    parts_h.append(f"earning the {', '.join(names)} badge{'s' if len(names) > 1 else ''}")
            if parts_h:
                chapters.append(". ".join(parts_h) + ".")

    # Chapter 4: Simulation resilience
    sim_data = LATEST_FED_SIM_RESULTS
    if sim_data and sim_data.get("ranked"):
        ranked = sim_data["ranked"]
        fed_avg = sim_data.get("fed_avg", 0)
        chapters.append(f"In simulation, average resilience across {len(ranked)} users is {fed_avg} percent confidence.")
        top_sim = ranked[0]
        chapters.append(f"{top_sim['user_id']} shows the highest resilience at {top_sim['avg_confidence']} percent with {top_sim['escalations']} escalations.")
    else:
        chapters.append("No simulation data available yet. Run a federated simulation to see resilience comparisons.")

    # Chapter 4: System health
    avg_conf = round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None
    if avg_conf is not None:
        chapters.append(f"System-wide average confidence is {avg_conf} percent across {len(RECENT_SCORES)} recent interactions.")
    mode = HYBRID_MODE
    chapters.append(f"System is in {mode} mode.")

    narration = " ".join(chapters)
    emotion = compute_narration_emotion(avg_conf=avg_conf, escalations=len(ESCALATION_EVENTS), consecutive_low=CONSECUTIVE_LOW)
    return {"narration": narration, "chapters": chapters, "emotion": emotion}


@app.get("/api/dynamic-boost")
async def get_dynamic_boost():
    return {"boost": DYNAMIC_BOOST, "consecutive_low": CONSECUTIVE_LOW, "avg_conf": round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None}


@app.get("/api/badge-alerts")
async def badge_alerts():
    global PREV_BADGE_MAP
    lb = await leaderboard()
    entries = lb.get("leaderboard", [])
    new_badges = []
    badge_names = {"\U0001F3C6": "Trophy", "\U0001F947": "Gold", "\U0001F948": "Silver", "\u2B50": "Star", "\U0001F504": "Syncer", "\u2705": "Clean", "\u26A1": "Energizer", "\U0001F3A4": "Narrator", "\U0001F9E9": "Strategist", "\U0001F91D": "Diplomat", "\U0001F31F": "Pioneer"}
    for e in entries:
        uid = e["user_id"]
        curr = set(e.get("badges", []))
        prev = set(PREV_BADGE_MAP.get(uid, []))
        gained = curr - prev
        for b in gained:
            nb = {"user_id": uid, "badge": b, "name": badge_names.get(b, b), "timestamp": datetime.now().isoformat()}
            new_badges.append(nb)
            BADGE_HISTORY.append(nb)
        PREV_BADGE_MAP[uid] = list(curr)
    if new_badges:
        parts = [f"{nb['user_id']} earned the {nb['name']} badge" for nb in new_badges]
        return {"alerts": new_badges, "narration": "Badge alert. " + ". ".join(parts) + "."}
    return {"alerts": [], "narration": None}


@app.get("/api/badge-history")
async def badge_history():
    return {"history": BADGE_HISTORY[-50:]}


@app.get("/api/weight-history")
async def weight_history():
    return {"history": WEIGHT_HISTORY[-50:], "score_history": RECENT_SCORES[-50:]}


@app.get("/api/orchestration-story")
async def orchestration_story():
    chapters = []
    recent_boosts = [w for w in WEIGHT_HISTORY[-10:] if w.get("boost", {}).get("changed")]
    if recent_boosts:
        last = recent_boosts[-1]["boost"]
        chapters.append(f"Orchestration boosted {last['persona']} by {last['boost']} points due to {last['reason']} via {last['source']} mode.")
        if len(recent_boosts) > 1:
            prev = recent_boosts[-2]["boost"]
            chapters.append(f"Earlier, {prev['persona']} was boosted for {prev['reason']}.")
    else:
        recent = [w for w in WEIGHT_HISTORY[-5:] if w.get("boost", {}).get("boost", 0) > 0]
        if recent:
            last = recent[-1]["boost"]
            chapters.append(f"Active persona boost is on {last['persona']} due to {last['reason']}.")
        else:
            chapters.append("No persona boosts active this session.")
    recent_badges = BADGE_HISTORY[-5:]
    if recent_badges:
        badge_parts = [f"{b['user_id']} earned {b['name']}" for b in recent_badges]
        chapters.append("Badges: " + ". ".join(badge_parts) + ".")
    recent_drift = [e for e in PERSONA_DRIFT_LOG[-5:] if e["user_id"] not in ("simulation", "federated")]
    if recent_drift:
        drift_parts = [f"{e['user_id']} triggered {e['action']}" for e in recent_drift]
        chapters.append("Drift events: " + ". ".join(drift_parts) + ".")
    mode = HYBRID_MODE
    chapters.append(f"Running in {mode} mode.")
    narration = "Session overview. " + " ".join(chapters)
    emotion = compute_narration_emotion(avg_conf=round(sum(RECENT_SCORES) / len(RECENT_SCORES), 1) if RECENT_SCORES else None, escalations=len(ESCALATION_EVENTS))
    if RECENT_SCORES:
        recent = RECENT_SCORES[-10:]
        if len(recent) >= 2:
            delta = round(recent[-1] - recent[0], 1)
            if abs(delta) >= 5:
                direction = "rose" if delta > 0 else "dropped"
                chapters.append(f"Confidence {direction} {abs(delta)} percent over the last {len(recent)} interactions.")
        if len(RECENT_SCORES) >= 5:
            recent5 = RECENT_SCORES[-5:]
            avg5 = sum(recent5) / len(recent5)
            spread = round(max(recent5) - min(recent5), 1)
            if spread <= 10:
                chapters.append(f"Confidence stability maintained over the last {len(recent5)} interactions with low variance.")
            elif spread > 25:
                chapters.append(f"Confidence volatility detected with a {spread} point spread in the last {len(recent5)} interactions.")
            healthy = sum(1 for s in recent5 if s >= 70)
            if healthy == len(recent5):
                chapters.append("All recent interactions show healthy confidence above 70 percent.")
            elif healthy == 0:
                chapters.append("All recent interactions show low confidence below 70 percent.")
    return {"narration": narration, "chapters": chapters, "emotion": emotion}


@app.get("/api/session-replay")
async def session_replay():
    story_resp = await orchestration_story()
    weight_data = WEIGHT_HISTORY[-50:]
    score_history = RECENT_SCORES[-20:]
    steps = []
    for i, w in enumerate(weight_data):
        step = {"index": i, "weights": w.get("weights", {}), "boost": w.get("boost", {}), "confidence": score_history[i] if i < len(score_history) else None}
        parts = []
        if w.get("boost", {}).get("changed"):
            b = w["boost"]
            parts.append(f"{b['persona']} boosted +{b['boost']} ({b['reason']})")
        step_narration = ". ".join(parts) if parts else f"Step {i}: weights {json.dumps(w.get('weights',{}))}"
        if i < len(score_history) and score_history[i] is not None:
            step_narration += f", confidence {score_history[i]}"
        step["narration"] = step_narration
        steps.append(step)
    return {
        "narration": story_resp["narration"],
        "chapters": story_resp["chapters"],
        "emotion": story_resp["emotion"],
        "weight_history": weight_data,
        "score_history": score_history,
        "badge_history": BADGE_HISTORY[-20:],
        "steps": steps,
    }


@app.get("/api/contributor-legacy")
async def contributor_legacy():
    users = set(e["user_id"] for e in PERSONA_DRIFT_LOG if e["user_id"] not in ("simulation", "federated"))
    badge_names = {"\U0001F3C6": "Trophy", "\U0001F947": "Gold", "\U0001F948": "Silver", "\u2B50": "Star", "\U0001F504": "Syncer", "\u2705": "Clean", "\u26A1": "Energizer", "\U0001F3A4": "Narrator", "\U0001F9E9": "Strategist", "\U0001F91D": "Diplomat", "\U0001F31F": "Pioneer"}
    legacy = []
    for uid in sorted(users):
        drift_events = [e for e in PERSONA_DRIFT_LOG if e["user_id"] == uid]
        user_badges = [b for b in BADGE_HISTORY if b["user_id"] == uid]
        weight_snaps = [e["weights"] for e in drift_events if e.get("weights")]
        weight_influence = {}
        for snap in weight_snaps:
            for p, v in snap.items():
                weight_influence[p] = v
        legacy.append({
            "user_id": uid,
            "total_events": len(drift_events),
            "last_action": drift_events[-1]["action"] if drift_events else None,
            "last_seen": drift_events[-1]["timestamp"] if drift_events else None,
            "badges": [{"badge": b["badge"], "name": b["name"], "timestamp": b["timestamp"]} for b in user_badges],
            "badge_count": len(user_badges),
            "weight_influence": weight_influence,
            "events_by_type": dict(Counter(e["action"] for e in drift_events)),
            "drift_timeline": [{"timestamp": e["timestamp"], "action": e["action"], "weights": e.get("weights", {})} for e in drift_events[-20:]],
            "badge_timeline": [{"timestamp": b["timestamp"], "badge": b["badge"], "name": b["name"]} for b in user_badges[-10:]],
        })
    return {"legacy": legacy}


@app.get("/api/contributor-legacy-narration")
async def contributor_legacy_narration():
    data = await contributor_legacy()
    sentences = []
    for u in data.get("legacy", []):
        parts = [f"Over {u['total_events']} events"]
        if u["badge_count"]:
            parts.append(f"{u['badge_count']} badges")
        if u.get("weight_influence"):
            boosts = [f"{p} plus {v}" for p, v in u["weight_influence"].items() if v > 0]
            if boosts:
                parts.append("influenced " + ", ".join(boosts))
        sentences.append(f"{u['user_id']}: " + ", ".join(parts) + ".")
        dt = u.get("drift_timeline", [])
        if len(dt) >= 3:
            for p in ("mentor", "coach", "teacher"):
                vals = [(i, e.get("weights", {}).get(p, 0)) for i, e in enumerate(dt) if e.get("weights", {}).get(p, 0) > 0]
                if vals:
                    peak = max(vals, key=lambda x: x[1])
                    if peak[1] >= 15:
                        when = dt[peak[0]].get("timestamp", "")
                        date_str = when[:10] if when else ""
                        peak_msg = f"{p} peaked at {peak[1]}"
                        if date_str:
                            peak_msg += f" around {date_str}"
                        bt = u.get("badge_timeline", [])
                        near_badge = [b for b in bt if abs(datetime.fromisoformat(b["timestamp"].replace("Z","")) - datetime.fromisoformat(when.replace("Z",""))).days <= 3 if when and b.get("timestamp")]
                        if near_badge:
                            peak_msg += f", earning {near_badge[0]['name']}"
                        sentences.append(f"For {u['user_id']}, " + peak_msg + ".")
        bt = u.get("badge_timeline", [])
        if bt:
            for b in bt:
                bd = b.get("timestamp", "")[:10]
                sentences.append(f"{u['user_id']} earned {b['name']} on {bd}.")
    narration = "Contributor legacy. " + " ".join(sentences)
    emotion = "encouragement" if any(u["badge_count"] > 0 for u in data.get("legacy", [])) else "reflection"
    return {"narration": narration, "emotion": emotion}


@app.get("/api/persona-influence")
async def persona_influence():
    cumulative = defaultdict(int)
    for e in PERSONA_DRIFT_LOG:
        if e.get("weights"):
            for p, v in e["weights"].items():
                cumulative[p] += v * 0.5
    return {"influence": dict(cumulative), "total": sum(cumulative.values())}


@app.get("/api/influence-history")
async def influence_history():
    history = []
    cumulative = defaultdict(int)
    for e in PERSONA_DRIFT_LOG:
        if e.get("weights"):
            for p, v in e["weights"].items():
                cumulative[p] += v * 0.5
        history.append({
            "timestamp": e.get("timestamp", ""),
            "cumulative": dict(cumulative),
            "total": sum(cumulative.values()),
        })
    return {"history": history[-50:]}


@app.get("/api/confidence-health")
async def confidence_health():
    if len(RECENT_SCORES) < 3:
        return {"status": "insufficient_data", "sessions": []}
    recent = RECENT_SCORES[-30:]
    sessions = []
    window = 5
    for i in range(0, len(recent), window):
        chunk = recent[i:i+window]
        if len(chunk) < 3:
            continue
        avg = round(sum(chunk) / len(chunk), 1)
        spread = round(max(chunk) - min(chunk), 1)
        healthy = sum(1 for s in chunk if s >= 70)
        sessions.append({
            "window": f"{i}-{i+len(chunk)-1}",
            "avg": avg,
            "spread": spread,
            "healthy_ratio": round(healthy / len(chunk), 2),
            "status": "stable" if spread <= 10 else "volatile" if spread > 25 else "moderate",
        })
    comparisons = []
    for i in range(1, len(sessions)):
        prev = sessions[i-1]
        cur = sessions[i]
        comparisons.append({
            "from_window": prev["window"],
            "to_window": cur["window"],
            "avg_delta": round(cur["avg"] - prev["avg"], 1),
            "spread_delta": round(cur["spread"] - prev["spread"], 1),
            "healthy_delta": round(cur["healthy_ratio"] - prev["healthy_ratio"], 2),
        })
    overall_spread = round(max(recent) - min(recent), 1)
    overall_avg = round(sum(recent) / len(recent), 1)
    overall_healthy = sum(1 for s in recent if s >= 70)
    first_avg = sessions[0]["avg"] if sessions else None
    last_avg = sessions[-1]["avg"] if sessions else None
    overall_delta = round(last_avg - first_avg, 1) if first_avg is not None and last_avg is not None else None
    return {
        "sessions": sessions,
        "comparisons": comparisons,
        "overall_delta": overall_delta,
        "overall": {
            "avg": overall_avg,
            "spread": overall_spread,
            "healthy_ratio": round(overall_healthy / len(recent), 2),
            "total_interactions": len(recent),
            "status": "stable" if overall_spread <= 10 else "volatile" if overall_spread > 25 else "moderate",
        },
        "trend": "improving" if len(sessions) >= 2 and sessions[-1]["avg"] > sessions[0]["avg"] else "declining" if len(sessions) >= 2 and sessions[-1]["avg"] < sessions[0]["avg"] else "stable",
    }


@app.get("/api/session-replay/export")
async def session_replay_export():
    story_resp = await orchestration_story()
    replay_resp = await session_replay()
    weight_data = WEIGHT_HISTORY[-50:]
    score_data = RECENT_SCORES[-20:]
    badge_data = BADGE_HISTORY[-20:]
    steps = replay_resp.get("steps", [])
    now_str = datetime.utcnow().isoformat()
    lines = [
        f"youfeel Session Report — {now_str}",
        f"Mode: {HYBRID_MODE}",
        "",
        "=== NARRATION ===",
        story_resp["narration"],
        "",
        "=== CHAPTERS ===",
    ]
    lines.extend(f"  {i+1}. {c}" for i, c in enumerate(story_resp["chapters"]))
    lines.extend(["", "=== TIMELINE EVENTS ==="])
    for s in steps:
        lines.append(f"  Step {s['index']}: {s['narration']}")
    lines.extend(["", "=== WEIGHT HISTORY ==="])
    for w in weight_data:
        w_str = json.dumps(w)
        lines.append(f"  {w_str}")
    lines.extend(["", "=== SCORE HISTORY ==="])
    lines.append(f"  {json.dumps(score_data)}")
    lines.extend(["", "=== BADGE HISTORY ==="])
    for b in badge_data:
        lines.append(f"  {b['user_id']} — {b['badge']} {b['name']} ({b['timestamp']})")
    text = "\n".join(lines)
    return Response(
        content=text,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="session-report-{now_str[:10]}.txt"'},
    )


@app.get("/api/contributor-legacy/export")
async def contributor_legacy_export():
    data = await contributor_legacy()
    narr_resp = await contributor_legacy_narration()
    now_str = datetime.utcnow().isoformat()
    lines = [
        f"youfeel Contributor Legacy Report — {now_str}",
        f"Mode: {HYBRID_MODE}",
        "",
        "=== NARRATION ===",
        narr_resp["narration"],
        "",
    ]
    for u in data.get("legacy", []):
        lines.append(f"--- {u['user_id']} ---")
        lines.append(f"  Total events: {u['total_events']}")
        lines.append(f"  Last action: {u['last_action']} ({u['last_seen'] or 'unknown'})")
        lines.append(f"  Badge count: {u['badge_count']}")
        for b in u.get("badges", []):
            lines.append(f"    {b['badge']} {b['name']} ({b['timestamp']})")
        lines.append(f"  Weight influence: {json.dumps(u.get('weight_influence', {}))}")
        lines.append(f"  Events by type: {json.dumps(u.get('events_by_type', {}))}")
        dt = u.get("drift_timeline", [])
        if dt:
            lines.append(f"  Drift timeline ({len(dt)} events):")
            for e in dt[-10:]:
                lines.append(f"    {e.get('timestamp','')} — {e['action']} {json.dumps(e.get('weights',{}))}")
        for p in ("mentor", "coach", "teacher"):
            vals = [(i, e.get("weights", {}).get(p, 0)) for i, e in enumerate(dt) if e.get("weights", {}).get(p, 0) > 0]
            if vals:
                peak = max(vals, key=lambda x: x[1])
                when = dt[peak[0]].get("timestamp", "")[:10] if dt[peak[0]].get("timestamp") else ""
                lines.append(f"  {p} peak: {peak[1]} at index {peak[0]}" + (f" around {when}" if when else ""))
        lines.append("")
    text = "\n".join(lines)
    return Response(
        content=text,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="contributor-legacy-{now_str[:10]}.txt"'},
    )


@app.get("/api/hybrid-mode")
async def get_hybrid_mode():
    cloud_key = bool(os.environ.get("LLM_CLOUD_KEY"))
    return {
        "mode": HYBRID_MODE,
        "cloud_configured": cloud_key,
        "local_url": LLM_API_URL,
        "local_model": LLM_MODEL,
        "cloud_url": os.environ.get("LLM_CLOUD_URL", "https://api.openai.com/v1/chat/completions"),
        "cloud_model": os.environ.get("LLM_CLOUD_MODEL", "gpt-4o-mini"),
    }


@app.post("/api/hybrid-mode")
async def set_hybrid_mode(data: dict):
    global HYBRID_MODE
    mode = data.get("mode", "local")
    if mode not in ("local", "cloud"):
        raise HTTPException(400, "Mode must be 'local' or 'cloud'")
    HYBRID_MODE = mode
    if mode == "local":
        global HYBRID_OFFLINE
        HYBRID_OFFLINE = False
    return {"mode": HYBRID_MODE, "status": "ok"}


@app.get("/api/hybrid-status")
async def hybrid_status():
    return {
        "mode": HYBRID_MODE,
        "offline": HYBRID_OFFLINE,
        "last_success": not HYBRID_OFFLINE if HYBRID_MODE == "cloud" else True,
    }


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
    for intent, total_count in sorted(INTENT_ESCALATION_COUNTER.items(), key=lambda x: -x[1]):
        key = f"intent_{intent}"
        if key in ESCALATION_REINFORCED:
            continue
        if total_count < 2:
            continue
        template = ESCALATION_TEMPLATES.get(intent, ESCALATION_TEMPLATES["general"])
        entry = {
            "intent": intent,
            "tier": "any",
            "escalation_count": total_count,
            "pattern": key,
            "prompt": template["prompt"],
            "completion": template["completion"],
            "generated_at": datetime.now().isoformat(),
        }
        FEWSHOT_DATASET.append(entry)
        ESCALATION_REINFORCED.add(key)
        ADAPTIVE_HISTORY.append({**entry, "type": "escalation_reinforce"})
        added.append(entry)
    COVERAGE_HISTORY.append({
        "timestamp": datetime.now().isoformat(),
        "total_fewshot": len(FEWSHOT_DATASET),
        "auto_reinforced": sorted(AUTO_REINFORCED_INTENTS),
        "escalation_reinforced": sorted(ESCALATION_REINFORCED),
        "intents_covered": sorted(set(e["intent"] for e in FEWSHOT_DATASET)),
        "escalation_count": sum(INTENT_ESCALATION_COUNTER.values()),
        "weak_intent_count": len([e for e in TRAINING_EVENTS if e["type"] == "weak_reply"]),
    })
    save_fewshot()
    if added:
        messages = [f"Escalation-reinforced '{a['intent']}' (pattern {a['pattern']}, {a['escalation_count']} events)." for a in added]
    else:
        messages = ["No escalation patterns to reinforce — all covered or below threshold."]
    bal = await auto_apply_balance()
    roll = await check_rollback()
    return {"reinforced": len(added), "total_fewshot": len(FEWSHOT_DATASET), "messages": messages, "added": [{"intent": a["intent"], "tier": a.get("tier", "any"), "count": a["escalation_count"]} for a in added], "balance_applied": bal.get("applied", False), "balance_adjustments": bal.get("adjustments"), "rollback_applied": roll.get("rolled_back", False), "rollback_reason": roll.get("reason")}


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
            reply, _ = generate_mock_reply(prompt, blend["persona"])
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
    avg_all = round(sum(r["avg_confidence"] for r in results) / len(results), 1) if results else None
    record_persona_drift("simulation", "simulate", sim_avg=avg_all)
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
        drift_logs = [e for e in PERSONA_DRIFT_LOG if e["user_id"] == uid and e["weights"]]
        last_weights = drift_logs[-1]["weights"] if drift_logs else None
        blends = DEFAULT_SIMULATION_BLENDS
        if last_weights:
            blends = [
                {"name": "User-weighted", "persona": "mentor:{}".format(
                    last_weights.get("mentor", 50)) + "+coach:{}".format(
                    last_weights.get("coach", 0)) + "+teacher:{}".format(
                    last_weights.get("teacher", 0)),
                    "weights": last_weights},
                {"name": "Balanced", "persona": "mentor:34+coach:33+teacher:33",
                 "weights": {"mentor": 34, "coach": 33, "teacher": 33}},
            ]
        elif c.get("sync_count", 0) > 0:
            blends = DEFAULT_SIMULATION_BLENDS[:3]
        scores = []
        total_escs = 0
        total_weaks = 0
        intents = Counter()
        blend_scores = []
        round_esc = []
        for blend in blends:
            b_scores = []
            for i in range(rounds):
                prompt = SIMULATION_PROMPTS[i % len(SIMULATION_PROMPTS)]
                reply, _ = generate_mock_reply(prompt, blend["persona"])
                score = score_reply(reply)
                b_scores.append(score)
                scores.append(score)
                intent = detect_intent(prompt)
                if intent:
                    intents[intent] += 1
                if score < 50:
                    total_weaks += 1
                triggered = score < 50 and len(scores) >= 3 and all(s < 50 for s in scores[-3:])
                if triggered:
                    total_escs += 1
                round_esc.append(1 if triggered else 0)
            blend_scores.append({"blend": blend["name"], "avg": round(sum(b_scores) / len(b_scores), 1) if b_scores else 0})
        total_prompts = rounds * len(blends)
        user_results.append({
            "user_id": uid,
            "sync_count": c.get("sync_count", 0),
            "conflict_count": c.get("conflict_count", 0),
            "weak_replies": USER_WEAK_REPLIES.get(uid, 0),
            "gold_exports": USER_GOLD_EXPORTS.get(uid, 0),
            "sim_avg_confidence": round(sum(scores) / len(scores), 1) if scores else 0,
            "sim_escalations": total_escs,
            "sim_escalation_rate": round(total_escs / total_prompts * 100, 1) if total_prompts else 0,
            "sim_weak_ratio": f"{round(total_weaks / total_prompts * 100, 1)}%" if total_prompts else "0%",
            "sim_weak_count": total_weaks,
            "sim_best_intent": intents.most_common(1)[0][0] if intents else "none",
            "blend_scores": blend_scores,
            "custom_weights": bool(last_weights),
            "round_esc_cumulative": round_esc,
        })
    ranked = sorted(user_results, key=lambda u: u["sim_avg_confidence"], reverse=True)
    fed_avg = round(sum(u["sim_avg_confidence"] for u in user_results) / len(user_results), 1) if user_results else None
    record_persona_drift("federated", "federated_simulate", sim_avg=fed_avg)
    result = {
        "results": user_results,
        "ranked": [
            {"rank": i + 1, "user_id": u["user_id"], "avg_confidence": u["sim_avg_confidence"],
             "escalations": u["sim_escalations"], "weak_ratio": u["sim_weak_ratio"]}
            for i, u in enumerate(ranked)
        ],
        "rounds": rounds,
        "total_users": len(users),
        "fed_avg": fed_avg,
    }
    global LATEST_FED_SIM_RESULTS
    LATEST_FED_SIM_RESULTS = result
    return result


@app.post("/api/gold-export")
async def record_gold_export(data: dict, req: Request):
    user_id = req.headers.get("X-User-Id") or "anonymous"
    count = data.get("count", 1)
    USER_GOLD_EXPORTS[user_id] += count
    record = {"user_id": user_id, "count": count, "timestamp": datetime.now().isoformat()}
    USER_ANALYTICS_HISTORY[user_id].append(record)
    if user_id in CONTRIBUTORS:
        CONTRIBUTORS[user_id]["gold_exports"] = USER_GOLD_EXPORTS[user_id]
    record_persona_drift(user_id, "gold_export")
    return {"status": "ok", "user_id": user_id, "total_gold": USER_GOLD_EXPORTS[user_id]}


@app.get("/api/persona-drift")
async def persona_drift():
    by_user = defaultdict(list)
    timeline = []
    for entry in PERSONA_DRIFT_LOG:
        tl = {
            "timestamp": entry["timestamp"],
            "user_id": entry["user_id"],
            "action": entry["action"],
            "sim_avg": entry["sim_avg"],
            "weights": entry["weights"],
        }
        timeline.append(tl)
        by_user[entry["user_id"]].append(tl)
    influence = {}
    for uid, logs in sorted(by_user.items()):
        weights_before = {}
        weights_after = {}
        for i, lg in enumerate(logs):
            if lg["weights"]:
                weights_after = dict(lg["weights"])
                if not weights_before:
                    weights_before = dict(lg["weights"])
        if not weights_before and not weights_after:
            continue
        drift = {}
        all_keys = set(weights_before.keys()) | set(weights_after.keys())
        for k in sorted(all_keys):
            before = weights_before.get(k, 50)
            after = weights_after.get(k, 50)
            diff = round(after - before, 1)
            if diff != 0:
                drift[k] = diff
        if drift or len(logs) > 0:
            influence[uid] = {
                "events": len(logs),
                "last_action": logs[-1]["action"] if logs else None,
                "weight_drift": drift,
                "total_sims": sum(1 for lg in logs if "simulate" in lg["action"]),
            }
    sim_avg_trend = [{"timestamp": e["timestamp"], "value": e["sim_avg"], "user_id": e["user_id"]} for e in PERSONA_DRIFT_LOG if e["sim_avg"] is not None]
    return {
        "timeline": timeline[-100:],
        "influence": influence,
        "sim_avg_trend": sim_avg_trend[-50:],
        "total_events": len(PERSONA_DRIFT_LOG),
    }


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
    record_persona_drift(user_id, "conflict_resolve")
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
