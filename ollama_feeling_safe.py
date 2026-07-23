import ollama
import json
import os
import sys
import shutil
import hashlib
import urllib.request
import urllib.parse
import re
import winsound
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time

MODEL_VERSION = "llama3.2"
DATASET_PATH = "curated_dataset.jsonl"
WEAK_REPLIES_PATH = "weak_replies.jsonl"
USER_ID = "anonymous"
CACHE_PATH = "knowledge_cache.jsonl"
CONFLICT_PATH = "conflict_queue.jsonl"
ESCALATION_LOG_PATH = "escalation_log.jsonl"
WEIGHT_LOG_PATH = "weight_history.jsonl"
CONFLICT_LOG_PATH = "conflict_resolution_log.jsonl"
GOLD_DIR = "gold"
SNAPSHOT_DIR = "snapshots"
SYNC_DIR = None
SCORE_THRESHOLD = 70
WEIGHT_BOOST_FACTOR = 12
WEIGHT_DECAY_PER_TURN = 3
DRIFT_THRESHOLD = 15
PREDICTIVE_BOOST = 8
CONFLICT_AUTO_POLICY = "low"
STALE_WARNING_DAYS = 1
AUTO_REFRESH_ENABLED = False
LAST_REPLY = None
LAST_PROMPT = None
LAST_SCORE = None
RECENT_SCORES = []
CONSECUTIVE_LOW = 0
DYNAMIC_WEIGHTS = {}
PERSONA_SPEC = "mentor"
CACHE_HITS = 0
CACHE_MISSES = 0
AUTO_RESOLVED_COUNT = 0
MANUAL_RESOLVED_COUNT = 0

VOICE_ENABLED = False
VOICE_RECOGNIZER = None
VOICE_ENGINE = None
VOICE_SPEECH_QUEUE = queue.Queue()
VOICE_PERSONA_VOICES = {
    "mentor": {"rate": 160, "volume": 0.9, "voice_name": "zira"},
    "coach": {"rate": 180, "volume": 0.85, "voice_name": "david"},
    "teacher": {"rate": 150, "volume": 0.9, "voice_name": "zira"},
}

EXTERNAL_ENABLED = False
EXTERNAL_SOURCES = {
    "duckduckgo": "https://api.duckduckgo.com/?q={query}&format=json&no_html=1",
    "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/summary/{query}",
}
FRESHNESS_DECAY = {"duckduckgo": 7, "wikipedia": 3}
FRESHNESS_STALE_AT = 30

PERSONAS = {
    "mentor": """You are a warm, empathetic mentor.
Always respond with encouragement, positivity, and emotional awareness.
Use supportive language, motivational tone, and show understanding.
Never give harmful or misleading advice.
If uncertain, say you don't know but offer to help explore it.""",
    "coach": """You are a direct, results-oriented coach.
Always respond with clarity, actionable steps, and accountability.
Push the user toward growth with firm but supportive language.
Never give harmful or misleading advice.
If uncertain, say you don't know but offer to help explore it.""",
    "teacher": """You are a patient, knowledgeable teacher.
Always respond with structured explanations, examples, and clarity.
Break complex topics into digestible steps.
Never give harmful or misleading advice.
If uncertain, say you don't know but offer to help explore it.""",
}

FALLBACK = "I don't have that information right now, but I can help you explore it."

PROMPT_TEMPLATES = {
    "explain": "Explain this in a clear, kind way:\n\n{{query}}",
    "motivate": "Give me a motivational push about:\n\n{{query}}",
    "advise": "Offer supportive advice on:\n\n{{query}}",
}

ESCALATION_EVENTS = []
BASELINE_WEIGHTS = {}


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_to_dataset(prompt, completion, path=DATASET_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")


def append_to_weak(prompt, completion, score, path=WEAK_REPLIES_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "completion": completion, "score": score}) + "\n")


def append_conflict(prompt, completion_a, score_a, completion_b, score_b):
    pri = topic_priority_for(prompt)
    entry = {"prompt": prompt, "completions": [{"completion": completion_a, "score": score_a}, {"completion": completion_b, "score": score_b}], "priority": pri, "flagged_at": datetime.now().isoformat()}
    with open(CONFLICT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"   Conflict flagged (priority: {pri})")


def log_escalation(tier, count, user_input):
    entry = {"tier": tier, "consecutive_low": count, "user_input": user_input[:100], "timestamp": datetime.now().isoformat()}
    ESCALATION_EVENTS.append(entry)
    with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_weights(weights):
    entry = {"weights": dict(weights), "timestamp": datetime.now().isoformat()}
    with open(WEIGHT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_conflict_resolution(method, prompt):
    entry = {"method": method, "prompt": prompt[:80], "timestamp": datetime.now().isoformat()}
    with open(CONFLICT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def query_hash(query):
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def freshness_score(source, cached_at_str):
    try:
        age_days = (datetime.now() - datetime.fromisoformat(cached_at_str)).days
    except Exception:
        return 0
    return max(0, 100 - age_days * FRESHNESS_DECAY.get(source, 5))


def time_until_stale(source, cached_at_str):
    try:
        age_days = (datetime.now() - datetime.fromisoformat(cached_at_str)).days
    except Exception:
        return 0
    decay = FRESHNESS_DECAY.get(source, 5)
    current = max(0, 100 - age_days * decay)
    if current <= FRESHNESS_STALE_AT: return 0
    return (current - FRESHNESS_STALE_AT) / decay


def cache_lookup(query):
    global CACHE_HITS, CACHE_MISSES
    if not os.path.exists(CACHE_PATH):
        CACHE_MISSES += 1; return None
    qh = query_hash(query)
    fresh, stale = [], []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            entry = json.loads(line)
            fs = freshness_score(entry.get("source", "duckduckgo"), entry.get("cached_at", ""))
            entry["_freshness"] = fs
            if fs < FRESHNESS_STALE_AT: stale.append(entry)
            else: fresh.append(entry)
    if stale:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            for entry in fresh:
                e = {k: v for k, v in entry.items() if not k.startswith("_")}
                f.write(json.dumps(e) + "\n")
    for entry in fresh:
        if entry.get("query_hash") == qh:
            CACHE_HITS += 1; return entry["results"]
    CACHE_MISSES += 1; return None


def cache_store(query, results):
    for source, _ in results:
        entry = {"query_hash": query_hash(query), "query": query, "source": source, "results": results, "cached_at": datetime.now().isoformat()}
        with open(CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def auto_refresh_cache():
    if not AUTO_REFRESH_ENABLED or not os.path.exists(CACHE_PATH):
        return 0
    with open(CACHE_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    refreshed = 0
    kept = []
    for e in entries:
        src = e.get("source", "duckduckgo")
        days_left = time_until_stale(src, e.get("cached_at", ""))
        if 0 < days_left <= STALE_WARNING_DAYS and EXTERNAL_ENABLED:
            q = e.get("query", "")
            if q:
                results = [fetch_source(src, q)]
                results = [r for r in results if r]
                if results:
                    cache_store(q, results)
                    refreshed += 1
                kept.append(e)
            else:
                kept.append(e)
        else:
            kept.append(e)
    if refreshed:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            for e in kept:
                entry = {k: v for k, v in e.items() if not k.startswith("_")}
                f.write(json.dumps(entry) + "\n")
        print(f"   Auto-refreshed {refreshed} cache entr(ies).")
    return refreshed


def forecast_cache_staleness():
    if not os.path.exists(CACHE_PATH): return
    with open(CACHE_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries: return
    warnings = []
    for e in entries:
        src = e.get("source", "duckduckgo")
        days_left = time_until_stale(src, e.get("cached_at", ""))
        if 0 < days_left <= STALE_WARNING_DAYS:
            warnings.append((src, e.get("query", "")[:60], days_left))
    if warnings:
        print(f"   Stale forecast ({len(warnings)} entries nearing staleness):")
        for src, q, d in warnings:
            print(f"      {src}: \"{q}\"... stale in {d:.1f}d")
        print("   Run a relevant query or enable AUTO_REFRESH_ENABLED.")


def generate_variations(prompt, corrected):
    vs = [corrected, corrected.replace("I think", "I believe"), corrected.replace("maybe", "").replace("perhaps", "")]
    for v in set(v for v in vs if len(v) > 20):
        append_to_dataset(prompt, v)


def score_reply(reply):
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


def score_completion(c):
    return score_reply(c)


def topic_distribution(data):
    topics = Counter()
    for p in [d["prompt"] for d in data]:
        matched = False
        for key in PROMPT_TEMPLATES:
            if p.lower().startswith(key): topics[key] += 1; matched = True; break
        if not matched: topics["general"] += 1
    return topics


def topic_priority_for(text):
    lower = text.lower()
    for words, score in [(["motivate", "inspire", "encourage", "push"], 90),
                          (["advise", "recommend", "suggest", "should"], 70),
                          (["explain", "what", "how", "define"], 50)]:
        if any(w in lower for w in words): return score
    return 30


def detect_intent(text):
    lower = text.lower()
    if any(w in lower for w in ["motivate", "inspire", "encourage", "push", "hype"]): return "motivate"
    if any(w in lower for w in ["explain", "what is", "how does", "define", "tell me about"]): return "explain"
    if any(w in lower for w in ["advise", "should i", "what should", "recommend", "suggest"]): return "advise"
    return None


def forecast_persona():
    if not os.path.exists(ESCALATION_LOG_PATH): return None
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: return None
    recent = events[-10:]
    intent_counts = Counter()
    for e in recent:
        text = e.get("user_input", ""); i = detect_intent(text)
        if i: intent_counts[i] += 1
    if not intent_counts: return None
    most_likely = intent_counts.most_common(1)[0][0]
    forecast = {"motivate": "coach:60+mentor:40", "explain": "teacher:60+coach:40", "advise": "mentor:50+coach:50"}
    result = forecast.get(most_likely)
    if result:
        print(f"   Forecast: next turn likely \"{most_likely}\" -> blend: {result}")
    return result


def predictive_intent_boost(intent):
    if not os.path.exists(ESCALATION_LOG_PATH): return {}
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: return {}
    by_intent = Counter()
    for e in events:
        text = e.get("user_input", ""); i = detect_intent(text) or "general"
        by_intent[i] += 1
    total = sum(by_intent.values())
    if total == 0: return {}
    rate = by_intent.get(intent, 0) / total
    boost_map = {}
    if rate > 0.3 and intent == "motivate":
        boost_map["coach"] = PREDICTIVE_BOOST; boost_map["mentor"] = PREDICTIVE_BOOST
    elif rate > 0.3 and intent == "explain":
        boost_map["teacher"] = PREDICTIVE_BOOST
    elif rate > 0.3 and intent == "advise":
        boost_map["mentor"] = PREDICTIVE_BOOST; boost_map["coach"] = int(PREDICTIVE_BOOST * 0.7)
    return boost_map


def reinforce_hotspots():
    if not os.path.exists(ESCALATION_LOG_PATH): return 0
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: return 0
    heat = Counter()
    for e in events:
        text = e.get("user_input", ""); i = detect_intent(text) or "general"
        heat[i] += 1
    riskiest = heat.most_common(1)
    if not riskiest: return 0
    intent = riskiest[0][0]
    count = riskiest[0][1]
    if count < 2: return 0
    reinforcement = {
        "motivate": "Motivate me to start a new habit. Respond with warmth and actionable steps.",
        "explain": "Explain how to stay consistent with daily goals. Be clear and structured.",
        "advise": "What should I do when I feel overwhelmed? Offer supportive advice.",
    }
    template = reinforcement.get(intent)
    if not template: return 0
    example_prompt = f"[auto-reinforcement] {template}"
    example_completion = f"This is an auto-generated reinforcement example for high-risk intent \"{intent}\" ({count} escalation events). Replace this with a high-quality curated response."
    append_to_dataset(example_prompt, example_completion)
    print(f"   Reinforced hotspot: added few-shot example for \"{intent}\" ({count} escalations).")
    return 1


def normalize_weights(w):
    if not w: return
    total = sum(w.values())
    if total == 0: return
    factor = 100 / total
    for k in w: w[k] = round(w[k] * factor)


def decay_weights(w):
    for k in w:
        if w[k] > 50: w[k] = max(50, w[k] - WEIGHT_DECAY_PER_TURN)


def orchestrate_persona(spec, overrides=None):
    parts = spec.split("+")
    segs = []
    for part in parts:
        part = part.strip()
        if ":" in part:
            name, ws = part.rsplit(":", 1); weight = int(ws) if ws.isdigit() else 50
        else: name = part; weight = 50
        base = PERSONAS.get(name.strip())
        if overrides and name.strip() in overrides: weight = overrides[name.strip()]
        if base: segs.append((base.strip(), weight))
    if not segs: return PERSONAS["mentor"] + f"\n\nRunning on {MODEL_VERSION}."
    tw = sum(w for _, w in segs) or 1
    parts_out = []
    for t, w in segs:
        pct = w / tw
        emp = "strongly" if pct > 0.5 else "moderately" if pct > 0.2 else "slightly"
        parts_out.append(f"[{emp} weighted at {w}%]\n{t}")
    return "\n\n".join(parts_out) + f"\n\nRunning on {MODEL_VERSION}."


def build_persona(data=None, persona_name="mentor", overrides=None):
    if "+" in persona_name: persona = orchestrate_persona(persona_name, overrides)
    else: persona = PERSONAS.get(persona_name, PERSONAS["mentor"])
    avg_c = (sum(RECENT_SCORES) / len(RECENT_SCORES)) if RECENT_SCORES else 100
    if avg_c < 50: persona += "\nBe cautious and acknowledge uncertainty clearly."
    elif avg_c < 70: persona += "\nBalance confidence with openness to correction."
    else: persona += "\nRespond with strong confidence and clarity."
    if data and len(data) >= 3:
        t = topic_distribution(data); tot = sum(t.values()) or 1
        if t.get("motivate", 0) / tot < 0.2: persona += "\nEmphasize motivation and encouragement."
        if t.get("advise", 0) / tot < 0.2: persona += "\nActively offer actionable advice."
        if t.get("explain", 0) / tot > 0.5: persona += "\nPrioritize clarity and simplicity."
    return persona


def topic_balance(data):
    topics = topic_distribution(data)
    if not topics: return topics
    total = sum(topics.values()); ideal = total / len(topics)
    print("   Balance:")
    for topic, count in topics.most_common():
        sign = "+" if count >= ideal else "-"
        print(f"      {topic}: {count} ({sign}{abs(count - ideal):.0f} vs ideal)")
    return topics


def show_cache_stats():
    if not os.path.exists(CACHE_PATH): print("   Cache: empty"); return
    with open(CACHE_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries: print("   Cache: empty"); return
    fs_by_src = defaultdict(list)
    for e in entries: fs_by_src[e.get("source", "unknown")].append(freshness_score(e.get("source", ""), e.get("cached_at", "")))
    tq = CACHE_HITS + CACHE_MISSES; hr = (CACHE_HITS / tq * 100) if tq else 0
    print(f"   Cache: {len(entries)} entries | hits: {CACHE_HITS} | misses: {CACHE_MISSES} | hit rate: {hr:.0f}%")
    for src, scores in fs_by_src.items():
        avg = sum(scores) / len(scores)
        bar = chr(9608) * int(avg / 10) + chr(9617) * (10 - int(avg / 10))
        print(f"      {src}: {avg:.0f}/100 {bar}")
    forecast_cache_staleness()
    refreshed = auto_refresh_cache()
    if refreshed: print(f"   Auto-refreshed {refreshed} stale entr(ies).")


def show_weight_trends():
    if not os.path.exists(WEIGHT_LOG_PATH): print("   Weight history: no data"); return
    with open(WEIGHT_LOG_PATH, encoding="utf-8") as f:
        history = [json.loads(line) for line in f if line.strip()]
    if not history: return
    names = list(history[0].get("weights", {}).keys())
    print(f"   Weight trends ({len(history)} snapshots):")
    for name in names:
        vals = [h["weights"].get(name, 0) for h in history]
        recent5 = vals[-5:]
        d = chr(8599) if len(vals) > 1 and vals[-1] > vals[0] else chr(8600) if len(vals) > 1 and vals[-1] < vals[0] else chr(8594)
        print(f"      {name}: {d} avg {sum(recent5)/len(recent5):.0f} last5: {recent5}")
    return history


def show_escalation_dashboard():
    if not os.path.exists(ESCALATION_LOG_PATH): print("   No escalations."); return
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: print("   Escalations: none"); return
    tiers = Counter(e.get("tier") for e in events)
    triggers = Counter(e.get("user_input", "")[:30] for e in events)
    total = len(events)
    labels = {1: "external lookup", 2: "persona shift", 3: "hard fallback"}
    print(f"   Escalations ({total} total):")
    for t in sorted(tiers):
        c = tiers[t]; p = c / total * 100
        bar = chr(9608) * int(p / 5) + chr(9617) * (20 - int(p / 5))
        print(f"      Tier {t} ({labels.get(t)}): {c} ({p:.0f}%) {bar}")
    print("      Top triggers:")
    for text, count in triggers.most_common(3):
        print(f"         \"{text}...\" x{count}")


def show_escalation_heatmap():
    if not os.path.exists(ESCALATION_LOG_PATH): print("   No data."); return
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: return
    heat = defaultdict(lambda: Counter())
    for e in events:
        text = e.get("user_input", ""); intent = detect_intent(text) or "general"
        heat[intent][e.get("tier", "?")] += 1
    print("   Heatmap (intent vs tier):")
    tiers = sorted({e.get("tier") for e in events})
    print("   " + "".join(f" T{t} " for t in tiers))
    for intent in sorted(heat):
        row = f"   {intent:10s}"
        for t in tiers:
            c = heat[intent].get(t, 0)
            row += f" {chr(9608) if c >= 3 else chr(9618) if c >= 1 else ' '} {c} "
        max_c = max(heat[intent].values()) if heat[intent] else 0
        print(f"{row} {chr(9608) * min(max_c, 10)}")
    risk = sorted(heat, key=lambda i: sum(heat[i].values()), reverse=True)
    if risk: print(f"   Hotspot: \"{risk[0]}\" ({sum(heat[risk[0]].values())} events)")


def show_predictive_analytics():
    if not os.path.exists(ESCALATION_LOG_PATH): print("   No data."); return
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: return
    weak = Counter()
    for e in events:
        text = e.get("user_input", ""); i = detect_intent(text) or "general"
        weak[i] += 1
    print("   Predictive analytics (high-risk):")
    for topic, count in weak.most_common():
        bar = chr(9608) * count + chr(9617) * (10 - count)
        print(f"      {topic}: {count} {bar}")
    if weak: print(f"   ! Pre-train on \"{weak.most_common(1)[0][0]}\" examples.")


def show_conflict_analytics():
    if not os.path.exists(CONFLICT_LOG_PATH):
        print("   No conflict resolution history."); return
    with open(CONFLICT_LOG_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    methods = Counter(e.get("method") for e in entries)
    total = len(entries)
    auto = methods.get("auto", 0)
    manual = methods.get("manual", 0)
    gui = methods.get("gui", 0)
    voice = methods.get("voice", 0)
    print(f"   Conflict resolutions ({total}): auto: {auto} ({auto/total*100:.0f}%) | manual: {manual} ({manual/total*100:.0f}%) | gui: {gui} ({gui/total*100:.0f}%) | voice: {voice} ({voice/total*100:.0f}%)")
    print(f"   Live counters: auto={AUTO_RESOLVED_COUNT} manual={MANUAL_RESOLVED_COUNT}")


def detect_drift():
    if not os.path.exists(WEIGHT_LOG_PATH): return
    with open(WEIGHT_LOG_PATH, encoding="utf-8") as f:
        history = [json.loads(line) for line in f if line.strip()]
    if len(history) < 3: return
    latest = history[-1].get("weights", {})
    print("   Drift detection:")
    for name in latest:
        vals = [h["weights"].get(name, 50) for h in history]
        bl, cur = vals[0], vals[-1]; drift = abs(cur - bl)
        if drift > DRIFT_THRESHOLD:
            d = "above" if cur > bl else "below"
            print(f"      !! {name}: drifted {drift:.0f} pts {d} baseline ({bl} -> {cur})")
        else: print(f"      OK {name}: stable ({bl} -> {cur}, drift {drift:.0f})")


def auto_resolve_conflicts():
    global AUTO_RESOLVED_COUNT
    cutoff = {"low": 50, "medium": 70, "all": 100}.get(CONFLICT_AUTO_POLICY, 50)
    if not os.path.exists(CONFLICT_PATH): return 0
    with open(CONFLICT_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries: return 0
    kept, resolved = [], 0
    for e in entries:
        pri = e.get("priority", 30)
        if pri < cutoff:
            ca, cb = e["completions"][0], e["completions"][1]
            sa, sb = ca.get("score", 0), cb.get("score", 0)
            if abs(sa - sb) >= 20: best = ca if sa > sb else cb
            elif sa >= 70 and sb >= 70: best = {"completion": ca["completion"] + "\n\n" + cb["completion"]}
            else: best = ca if sa >= sb else cb
            append_to_dataset(e["prompt"], best["completion"])
            log_conflict_resolution("auto", e["prompt"])
            AUTO_RESOLVED_COUNT += 1; resolved += 1
        else: kept.append(e)
    if kept:
        with open(CONFLICT_PATH, "w", encoding="utf-8") as f:
            for e in kept: f.write(json.dumps(e) + "\n")
    elif os.path.exists(CONFLICT_PATH): os.remove(CONFLICT_PATH)
    return resolved


def show_dashboard():
    print("=" * 58)
    print("  EMOTIONAL OLLAMA DASHBOARD")
    print("=" * 58)
    data = load_dataset()
    if data: print(f"\nDataset: {len(data)} entries"); topic_balance(data)
    else: print("\nDataset: empty")
    print(); show_cache_stats(); print()
    hist = show_weight_trends(); print()
    if hist and len(hist) >= 3: detect_drift(); print()
    show_escalation_dashboard(); print()
    show_escalation_heatmap(); print()
    show_predictive_analytics(); print()
    show_conflict_analytics(); print()
    print("   Persona forecast:"); forecast_persona(); print()
    resolved = auto_resolve_conflicts()
    if resolved: print(f"   Auto-resolved {resolved} conflict(s) (policy: {CONFLICT_AUTO_POLICY}).")
    r = reinforce_hotspots()
    if r: print(f"   Added reinforcement example.")
    show_conflicts()
    print("=" * 58)


def dataset_stats():
    data = load_dataset()
    if not data: print("Dataset is empty."); return
    topic_balance(data)
    print(f"Dataset: {len(data)} entries | Model: {MODEL_VERSION}")
    avg_c = (sum(RECENT_SCORES) / len(RECENT_SCORES)) if RECENT_SCORES else None
    if avg_c is not None: print(f"   Avg confidence: {avg_c:.0f}/100 | Consecutive low: {CONSECUTIVE_LOW}")
    show_cache_stats(); show_weight_trends(); show_escalation_dashboard(); show_escalation_heatmap(); show_predictive_analytics(); show_conflict_analytics(); detect_drift()


def show_conflicts():
    if not os.path.exists(CONFLICT_PATH): print("   No conflicts."); return
    with open(CONFLICT_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    entries.sort(key=lambda e: e.get("priority", 30), reverse=True)
    print(f"   Conflicts ({len(entries)}):")
    for i, e in enumerate(entries):
        pp = e.get("prompt", "")[:60]
        sc = [c.get("score", 0) for c in e.get("completions", [])]
        print(f"   [{i}] [pri {e.get('priority', 30)}] {pp}... scores={sc}")
    print("   resolve <i> <text> | resolve-skip <i>")


def resolve_conflict(idx, correction, path=CONFLICT_PATH, method="manual"):
    global MANUAL_RESOLVED_COUNT
    if not os.path.exists(path): print("No conflict queue."); return
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if idx < 0 or idx >= len(entries): print(f"Invalid index {idx}."); return
    prompt = entries[idx]["prompt"]; append_to_dataset(prompt, correction)
    log_conflict_resolution(method, prompt)
    if method == "gui": pass
    elif method == "voice": pass
    else: MANUAL_RESOLVED_COUNT += 1
    print(f"Conflict [{idx}] resolved via {method}."); entries.pop(idx)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")
    if correction: generate_variations(prompt, correction)


def skip_conflict(idx, path=CONFLICT_PATH):
    if not os.path.exists(path): print("No conflict queue."); return
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if idx < 0 or idx >= len(entries): print(f"Invalid index {idx}."); return
    entries.pop(idx)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")
    print(f"Skipped conflict [{idx}].")


def validate_dataset(path=DATASET_PATH):
    data = load_dataset(path)
    if not data: print("No entries."); return
    issues = 0
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "prompt" not in entry or "completion" not in entry:
            print(f"Entry {i}: malformed"); issues += 1; continue
        if len(entry["completion"]) < 5: print(f"Entry {i}: too short"); issues += 1
        if "i don't know" in entry["completion"].lower() and len(entry["completion"]) < 20:
            print(f"Entry {i}: weak uncertain reply"); issues += 1
    print(f"{'All valid.' if issues == 0 else f'{issues} issue(s).'}")


def export_snapshot(path=DATASET_PATH, label=""):
    if not os.path.exists(path): print("No dataset."); return
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = f"_{label}" if label else ""
    sp = os.path.join(SNAPSHOT_DIR, f"curated_{ts}{suf}.jsonl")
    shutil.copy2(path, sp); print(f"Snapshot: {sp}"); return sp


def export_gold_dataset(path=DATASET_PATH, min_score=90):
    data = load_dataset(path)
    if not data: print("No entries."); return
    os.makedirs(GOLD_DIR, exist_ok=True)
    gp = os.path.join(GOLD_DIR, "gold_dataset.jsonl"); count = 0
    with open(gp, "w", encoding="utf-8") as out:
        for entry in data:
            if score_completion(entry["completion"]) >= min_score:
                out.write(json.dumps({"prompt": entry["prompt"], "completion": entry["completion"], "_score": score_completion(entry["completion"])}) + "\n"); count += 1
    print(f"Gold: {gp} ({count} entries)"); return gp


def sync_dataset(target_dir=None, user_id="anonymous"):
    dest = target_dir or SYNC_DIR
    if not dest: print("No sync target."); return
    os.makedirs(dest, exist_ok=True)
    tag_path = os.path.join(dest, "CONTRIBUTORS.txt")
    with open(tag_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {user_id}\n")
    files = [f for f in [DATASET_PATH, WEAK_REPLIES_PATH, CONFLICT_PATH, CONFLICT_LOG_PATH, ESCALATION_LOG_PATH, WEIGHT_LOG_PATH] if os.path.exists(f)]
    s = export_snapshot(); g = export_gold_dataset()
    if s: files.append(s)
    if g: files.append(g)
    for f in files: shutil.copy2(f, os.path.join(dest, os.path.basename(f))); print(f"   Synced: {os.path.basename(f)}")
    print(f"{user_id} sync -> {dest}")
    return True


def federated_sync(remote_dir, merge_policy="highest"):
    if not os.path.isdir(remote_dir): print(f"Remote not found: {remote_dir}"); return
    local = load_dataset()
    rp = os.path.join(remote_dir, DATASET_PATH)
    remote = load_dataset(rp) if os.path.exists(rp) else []
    seen, conflicts = {}, []
    for entry in local + remote:
        key = entry.get("prompt", "")
        if key in seen:
            es = score_completion(seen[key].get("completion", ""))
            cs = score_completion(entry.get("completion", ""))
            if merge_policy in ("keep-both", "manual"):
                conflicts.append((key, seen[key]["completion"], es, entry["completion"], cs)); continue
            else:
                if cs > es: seen[key] = entry
                elif cs == es: conflicts.append((key, seen[key]["completion"], es, entry["completion"], cs))
        else: seen[key] = entry
    for key, ca, sa, cb, sb in conflicts:
        if merge_policy == "keep-both":
            seen[key + "_v1"] = {"prompt": key + " [v1]", "completion": ca}
            seen[key + "_v2"] = {"prompt": key + " [v2]", "completion": cb}
        elif merge_policy == "manual": append_conflict(key, ca, sa, cb, sb)
    merged = list(seen.values())
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for entry in merged: f.write(json.dumps({"prompt": entry["prompt"], "completion": entry["completion"]}) + "\n")
    print(f"Merge [{merge_policy}]: {len(merged)} entries ({len(local)} local + {len(remote)} remote)")


def fetch_source(source_name, query):
    url = EXTERNAL_SOURCES.get(source_name)
    if not url: return None
    try:
        full = url.format(query=query.strip().replace(" ", "_")) if source_name == "wikipedia" else url.format(query=urllib.parse.quote(query))
        req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if source_name == "duckduckgo":
                ab = data.get("AbstractText", "")
                if ab: return (source_name, ab)
                rs = data.get("RelatedTopics", [])
                if rs and rs[0].get("Text"): return (source_name, rs[0]["Text"])
            elif source_name == "wikipedia":
                ext = data.get("extract", "")
                if ext: return (source_name, ext[:500])
    except Exception: pass
    return None


def layered_lookup(query):
    cached = cache_lookup(query)
    if cached: return cached
    results = [r for src in EXTERNAL_SOURCES if (r := fetch_source(src, query))]
    if results: cache_store(query, results)
    return results


def apply_template(user_input, dynamic_weights=None):
    intent = detect_intent(user_input) if dynamic_weights is not None else None
    if intent and dynamic_weights is not None:
        decay_weights(dynamic_weights)
        boost = predictive_intent_boost(intent)
        if intent == "motivate":
            for n in ["coach", "mentor"]:
                if n in dynamic_weights: dynamic_weights[n] = min(dynamic_weights.get(n, 50) + WEIGHT_BOOST_FACTOR + boost.get(n, 0), 100)
        elif intent == "explain" and "teacher" in dynamic_weights:
            dynamic_weights["teacher"] = min(dynamic_weights["teacher"] + WEIGHT_BOOST_FACTOR + boost.get("teacher", 0), 100)
        elif intent == "advise":
            for n in ["mentor", "coach"]:
                if n in dynamic_weights: dynamic_weights[n] = min(dynamic_weights.get(n, 50) + int(WEIGHT_BOOST_FACTOR * 0.8) + boost.get(n, 0), 100)
        normalize_weights(dynamic_weights)
    for key, template in PROMPT_TEMPLATES.items():
        if user_input.lower().startswith(key): return template.format(query=user_input[len(key):].strip())
    return user_input


def train_from_dataset(base_model=MODEL_VERSION, path=DATASET_PATH, new_model="empathetic-mentor", min_score=80):
    data = load_dataset(path)
    if len(data) < 5: print("Need >= 5."); return
    scored = sorted([(score_completion(e["completion"]), e) for e in data], key=lambda x: x[0], reverse=True)
    hq = [e for s, e in scored if s >= min_score] or [e for _, e in scored[:10]]
    export_snapshot(path)
    print(f"Training '{new_model}' from {base_model} ({len(hq)} examples)...")
    few_shot = "\n\n".join(f"User: {e['prompt']}\nAssistant: {e['completion']}" for e in hq[:20])
    persona = build_persona(data)
    mf = f"""FROM {base_model}
SYSTEM \"\"\"{persona.strip()}\"\"\"
TEMPLATE \"\"\"System: {persona.strip()}

{few_shot}
User: {{.Prompt}}
Assistant: \"\"\"
"""
    mp = "Modelfile.tmp"
    with open(mp, "w", encoding="utf-8") as f: f.write(mf)
    try:
        ollama.create(model=new_model, modelfile=mp)
        print(f"Model '{new_model}' created.")
    except Exception as e: print(f"Training failed: {e}")
    finally:
        if os.path.exists(mp): os.remove(mp)


def handle_escalation_tier(count, user_input, prompt):
    if count >= 7:
        print(f"   Tier 3 ({count}) - hard fallback"); log_escalation(3, count, user_input)
        return "I'm not confident I can give a reliable answer right now. Let's try a different approach."
    if count >= 5:
        print(f"   Tier 2 ({count}) - persona shift"); log_escalation(2, count, user_input); return None
    if count >= 3:
        print(f"   Tier 1 ({count}) - external lookup"); log_escalation(1, count, user_input)
        results = layered_lookup(user_input) if EXTERNAL_ENABLED else []
        if results: return "I checked multiple sources:\n" + "\n".join(f"[{s}] {t}" for s, t in results)
        return None
    return None


NARRATION_INTERVAL = 0
_last_drift_count = 0

def start_narration_scheduler(interval_minutes):
    def _loop():
        global _last_drift_count
        _narration_cycle = 0
        while True:
            time.sleep(interval_minutes * 60)
            if VOICE_ENGINE:
                speak_analytics_trends()
            _narration_cycle += 1
            if _narration_cycle % 2 == 0:
                speak_text(f"Current user: {USER_ID}. Sync stats for this session.")
            if os.path.exists(WEIGHT_LOG_PATH):
                with open(WEIGHT_LOG_PATH, encoding="utf-8") as f:
                    history = [json.loads(line) for line in f if line.strip()]
                drift_count = 0
                if len(history) >= 3:
                    latest = history[-1].get("weights", {})
                    for name in latest:
                        vals = [h["weights"].get(name, 50) for h in history]
                        if abs(vals[-1] - vals[0]) > DRIFT_THRESHOLD:
                            drift_count += 1
                if drift_count > _last_drift_count and drift_count > 0:
                    speak_text(f"Alert: {drift_count} persona drift events detected.")
                _last_drift_count = drift_count
    threading.Thread(target=_loop, daemon=True).start()


TRAINING_EVENTS = []  # list of dicts: {type, intent, score, count, timestamp}


def speak_training_narration():
    if not TRAINING_EVENTS:
        speak_text("No training events recorded yet.")
        return
    weak = [e for e in TRAINING_EVENTS if e["type"] == "weak_reply"]
    reinforces = [e for e in TRAINING_EVENTS if e["type"] == "reinforce"]
    parts = []
    if weak:
        parts.append(f"Total weak replies: {len(weak)}. Most recent score: {weak[-1]['score']}.")
    if reinforces:
        parts.append(f"Reinforcements applied: {len(reinforces)}. Last reinforced: {reinforces[-1]['intent']}.")
    if not parts:
        parts.append("No significant training events.")
    parts.append(f"Total training events tracked: {len(TRAINING_EVENTS)}.")
    summary = " ".join(parts)
    print(f"   Training narration: {summary}")
    speak_text(summary)


def speak_analytics_trends():
    parts = []
    if TRAINING_EVENTS:
        weak = [e for e in TRAINING_EVENTS if e["type"] == "weak_reply"]
        parts.append(f"Training: {len(weak)} weak replies out of {len(TRAINING_EVENTS)} events.")
    if not os.path.exists(WEIGHT_LOG_PATH):
        speak_text("No weight history available.")
        return
    with open(WEIGHT_LOG_PATH, encoding="utf-8") as f:
        history = [json.loads(line) for line in f if line.strip()]
    if len(history) < 3:
        speak_text("Not enough history for trend analysis.")
        return
    latest = history[-1].get("weights", {})
    for name in latest:
        vals = [h["weights"].get(name, 50) for h in history]
        bl, cur = vals[0], vals[-1]
        diff = cur - bl
        if abs(diff) > DRIFT_THRESHOLD:
            direction = "drifted up" if diff > 0 else "drifted down"
            parts.append(f"{name} persona has {direction} {abs(diff):.0f} points from baseline.")
        elif abs(diff) > 5:
            direction = "increased" if diff > 0 else "decreased"
            parts.append(f"{name} persona {direction} by {abs(diff):.0f} points.")
    if not parts:
        parts.append("All personas are stable.")
    summary = " ".join(parts)
    print(f"   Voice trends: {summary}")
    speak_text(summary)
    return parts


def dashboard_gui():
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter not available. Cannot start GUI.")
        return

    root = tk.Tk()
    root.title("Ollama Dashboard")
    root.geometry("780x750")
    root.configure(bg="#1e1e2e")

    title = tk.Label(root, text="Emotional Ollama Dashboard", font=("Segoe UI", 14, "bold"),
                     fg="#cdd6f4", bg="#1e1e2e")
    title.pack(pady=8)

    main_frame = tk.Frame(root, bg="#1e1e2e")
    main_frame.pack(fill="both", expand=True, padx=10, pady=5)

    notepad = ttk.Notebook(main_frame)
    notepad.pack(fill="both", expand=True)

    tab_charts = tk.Frame(notepad, bg="#1e1e2e")
    tab_conflicts = tk.Frame(notepad, bg="#1e1e2e")
    tab_controls = tk.Frame(notepad, bg="#1e1e2e")
    notepad.add(tab_charts, text="Charts")
    notepad.add(tab_conflicts, text="Conflicts")
    notepad.add(tab_controls, text="Controls")

    weight_frame = tk.LabelFrame(tab_charts, text="Persona Weights", fg="#cdd6f4",
                                  bg="#1e1e2e", font=("Segoe UI", 10, "bold"))
    weight_frame.pack(fill="x", pady=4)
    weight_canvas = tk.Canvas(weight_frame, height=80, bg="#1e1e2e", highlightthickness=0)
    weight_canvas.pack(fill="x", padx=5, pady=5)

    cache_frame = tk.LabelFrame(tab_charts, text="Cache Freshness", fg="#cdd6f4",
                                 bg="#1e1e2e", font=("Segoe UI", 10, "bold"))
    cache_frame.pack(fill="x", pady=4)
    cache_canvas = tk.Canvas(cache_frame, height=60, bg="#1e1e2e", highlightthickness=0)
    cache_canvas.pack(fill="x", padx=5, pady=5)

    heat_frame = tk.LabelFrame(tab_charts, text="Escalation Heatmap", fg="#cdd6f4",
                                bg="#1e1e2e", font=("Segoe UI", 10, "bold"))
    heat_frame.pack(fill="both", expand=True, pady=4)
    heat_canvas = tk.Canvas(heat_frame, bg="#1e1e2e", highlightthickness=0)
    heat_canvas.pack(fill="both", expand=True, padx=5, pady=5)

    conflict_canvas = tk.Canvas(tab_conflicts, bg="#1e1e2e", highlightthickness=0)
    conflict_canvas.pack(fill="both", expand=True, padx=5, pady=5)
    conflict_inner = tk.Frame(conflict_canvas, bg="#1e1e2e")
    conflict_canvas.create_window((0, 0), window=conflict_inner, anchor="nw")

    def _on_conflict_configure(event):
        conflict_canvas.configure(scrollregion=conflict_canvas.bbox("all"))
    conflict_inner.bind("<Configure>", _on_conflict_configure)

    status_bar = tk.Label(root, text="", fg="#a6adc8", bg="#1e1e2e",
                          font=("Segoe UI", 9), anchor="w")
    status_bar.pack(fill="x", padx=10, pady=2)

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(pady=5)
    speak_btn = tk.Button(btn_frame, text="Speak Summary", command=speak_dashboard_summary,
                          bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
                          relief="flat", padx=10)
    speak_btn.pack(side="left", padx=4)
    trends_btn = tk.Button(btn_frame, text="Analyze Trends", command=speak_analytics_trends,
                           bg="#f9e2af", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
                           relief="flat", padx=10)
    trends_btn.pack(side="left", padx=4)

    slider_vars = {}
    for pname in ["mentor", "coach", "teacher"]:
        row = tk.Frame(tab_controls, bg="#1e1e2e")
        row.pack(fill="x", pady=6, padx=10)
        c = {"mentor": "#89b4fa", "coach": "#a6e3a1", "teacher": "#f9e2af"}.get(pname, "#89b4fa")
        tk.Label(row, text=f" {pname}", bg="#1e1e2e", fg=c, font=("Segoe UI", 10, "bold"),
                 width=10, anchor="w").pack(side="left")
        slider_vars[pname] = tk.IntVar(value=DYNAMIC_WEIGHTS.get(pname, 50))
        slider = tk.Scale(row, from_=0, to=100, orient="horizontal", variable=slider_vars[pname],
                          bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244", highlightthickness=0,
                          length=300, font=("Segoe UI", 8))
        slider.pack(side="left", padx=5)
        val_label = tk.Label(row, textvariable=slider_vars[pname], bg="#1e1e2e", fg="#cdd6f4",
                             font=("Segoe UI", 9, "bold"), width=4)
        val_label.pack(side="left")

    def apply_sliders():
        changed = False
        for pname, var in slider_vars.items():
            new_val = var.get()
            if DYNAMIC_WEIGHTS.get(pname, 50) != new_val:
                DYNAMIC_WEIGHTS[pname] = new_val
                changed = True
        if changed:
            if VOICE_ENGINE:
                apply_persona_voice(PERSONA_SPEC)
            normalize_weights(DYNAMIC_WEIGHTS)
            log_weights(DYNAMIC_WEIGHTS)
            print(f"   Slider weights: {DYNAMIC_WEIGHTS}")

    apply_btn = tk.Button(tab_controls, text="Apply Weights", command=apply_sliders,
                          bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
                          relief="flat", padx=10)
    apply_btn.pack(pady=10)

    reset_btn = tk.Button(tab_controls, text="Reset to 50", command=lambda: [
        slider_vars[p].set(50) for p in slider_vars],
                          bg="#f38ba8", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
                          relief="flat", padx=10)
    reset_btn.pack(pady=2)

    COLORS = {"mentor": "#89b4fa", "coach": "#a6e3a1", "teacher": "#f9e2af"}
    AVATARS = {"mentor": "M", "coach": "C", "teacher": "T"}
    _anim_targets = {}

    def draw_animated_bars(canvas, items, color_map, max_val=100, tag="bars"):
        nonlocal _anim_targets
        w = canvas.winfo_width() or 400
        h = canvas.winfo_height() or 80
        canvas.delete(tag)
        if not items:
            canvas.create_text(w // 2, h // 2, text="No data", fill="#6c7086",
                               font=("Segoe UI", 9), tags=tag)
            return
        n = len(items)
        bar_w = max(24, (w - 20) // n - 12)
        targets = {}
        for i, (label, val) in enumerate(items):
            x0 = 15 + i * (bar_w + 12)
            bh = max(4, val / max_val * (h - 30))
            y0 = h - 15 - bh
            x1 = x0 + bar_w
            targets[label] = (x0, y0, x1, bh)
        _anim_targets[tag] = (_anim_targets.get(tag, {}), targets)

        def animate(step=0, max_steps=10):
            old_targets, new_targets = _anim_targets.get(tag, ({}, {}))
            if step >= max_steps:
                _draw_final(canvas, new_targets, color_map, h, bar_w, tag)
                return
            t = step / max_steps
            interp = {}
            for label in new_targets:
                old = old_targets.get(label)
                new = new_targets[label]
                if old:
                    ix0 = old[0] + (new[0] - old[0]) * t
                    iy0 = old[1] + (new[1] - old[1]) * t
                    ix1 = old[2] + (new[2] - old[2]) * t
                    ibh = old[3] + (new[3] - old[3]) * t
                else:
                    ix0, iy0, ix1, ibh = new
                    iy0 = h - 15
                    ibh = 4
                interp[label] = (ix0, iy0, ix1, ibh)
            _draw_final(canvas, interp, color_map, h, bar_w, tag)
            canvas.after(40, animate, step + 1, max_steps)

        animate()

    def _draw_final(canvas, items_dict, color_map, h, bar_w, tag):
        canvas.delete(tag)
        for label, (x0, y0, x1, bh) in items_dict.items():
            color = color_map.get(label, "#89b4fa")
            canvas.create_rectangle(x0, y0, x1, h - 15, fill=color, outline="", tags=tag)
            cx = (x0 + x1) // 2
            avatar = AVATARS.get(label, "?")
            canvas.create_oval(cx - 8, y0 - 16, cx + 8, y0, fill=color, outline="#45475a", tags=tag)
            canvas.create_text(cx, y0 - 8, text=avatar, fill="#1e1e2e",
                               font=("Segoe UI", 8, "bold"), tags=tag)
            canvas.create_text(cx, h - 4, text=f"{label} {int(bh * 100 / (h - 30) * 100 / 100) if (h - 30) else 0}",
                               fill="#cdd6f4", font=("Segoe UI", 7), tags=tag)

    def draw_heatmap():
        heat_canvas.delete("heat")
        w = heat_canvas.winfo_width() or 400
        h = heat_canvas.winfo_height() or 150
        events = []
        if os.path.exists(ESCALATION_LOG_PATH):
            with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
        if not events:
            heat_canvas.create_text(w // 2, h // 2, text="No escalation data",
                                    fill="#6c7086", font=("Segoe UI", 9), tags="heat")
            return
        heat = defaultdict(lambda: Counter())
        for e in events:
            text = e.get("user_input", "")
            intent = detect_intent(text) or "general"
            heat[intent][e.get("tier", "?")] += 1
        intents = sorted(heat)
        tiers = sorted({e.get("tier") for e in events})
        if not intents or not tiers:
            return
        cell_w = max(30, (w - 40) // len(tiers))
        cell_h = max(24, (h - 40) // len(intents))
        max_c = max((heat[i][t] for i in intents for t in tiers), default=1)
        for ri, intent in enumerate(intents):
            heat_canvas.create_text(10, 15 + ri * (cell_h + 4) + cell_h // 2,
                                    text=intent[:8], fill="#cdd6f4",
                                    font=("Segoe UI", 8), anchor="w", tags="heat")
            for ci, tier in enumerate(tiers):
                x0 = 80 + ci * (cell_w + 4)
                y0 = 10 + ri * (cell_h + 4)
                count = heat[intent].get(tier, 0)
                intensity = min(1.0, count / max_c) if max_c else 0
                r = int(30 + (205 - 30) * intensity)
                g = int(30 + (180 - 30) * (1 - intensity))
                b = int(30 + (220 - 30) * (1 - intensity))
                color = f"#{r:02x}{g:02x}{b:02x}"
                heat_canvas.create_rectangle(x0, y0, x0 + cell_w, y0 + cell_h,
                                              fill=color, outline="#45475a", tags="heat")
                heat_canvas.create_text(x0 + cell_w // 2, y0 + cell_h // 2,
                                        text=str(count), fill="#cdd6f4" if count else "#6c7086",
                                        font=("Segoe UI", 8), tags="heat")
            for ci, tier in enumerate(tiers):
                heat_canvas.create_text(80 + ci * (cell_w + 4) + cell_w // 2, 0,
                                        text=f"T{tier}", fill="#a6adc8",
                                        font=("Segoe UI", 8), anchor="n", tags="heat")

    def rebuild_conflicts():
        for w in conflict_inner.winfo_children():
            w.destroy()
        if not os.path.exists(CONFLICT_PATH):
            tk.Label(conflict_inner, text="No conflicts.", bg="#1e1e2e", fg="#6c7086",
                     font=("Segoe UI", 9)).pack(pady=20)
            return
        with open(CONFLICT_PATH, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        if not entries:
            tk.Label(conflict_inner, text="No conflicts.", bg="#1e1e2e", fg="#6c7086",
                     font=("Segoe UI", 9)).pack(pady=20)
            return
        entries.sort(key=lambda e: e.get("priority", 30), reverse=True)
        for idx, e in enumerate(entries):
            prompt = e.get("prompt", "")[:60]
            sc = [c.get("score", 0) for c in e.get("completions", [])]
            row = tk.Frame(conflict_inner, bg="#313244", bd=1, relief="solid")
            row.pack(fill="x", pady=2, padx=2)
            tk.Label(row, text=f"[{idx}] pri {e.get('priority', 30)}",
                     bg="#313244", fg="#a6adc8", font=("Segoe UI", 8, "bold"), width=12).pack(side="left", padx=4)
            tk.Label(row, text=f"{prompt}...  scores={sc}",
                     bg="#313244", fg="#cdd6f4", font=("Segoe UI", 8)).pack(side="left", fill="x", expand=True)
            def resolve_cb(i=idx, entry=e):
                ca = entry["completions"][0]
                cb = entry["completions"][1]
                sa, sb = ca.get("score", 0), cb.get("score", 0)
                best = ca if sa >= sb else cb
                resolve_conflict(i, best["completion"], method="gui")
                rebuild_conflicts()
                if VOICE_ENGINE:
                    speak_text("Conflict resolved.")
            def skip_cb(i=idx):
                skip_conflict(i)
                rebuild_conflicts()
                if VOICE_ENGINE:
                    speak_text("Conflict skipped.")
            tk.Button(row, text="Resolve", command=resolve_cb,
                      bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 8, "bold"),
                      relief="flat", padx=6).pack(side="right", padx=2)
            tk.Button(row, text="Skip", command=skip_cb,
                      bg="#f38ba8", fg="#1e1e2e", font=("Segoe UI", 8, "bold"),
                      relief="flat", padx=6).pack(side="right", padx=2)
        conflict_canvas.configure(scrollregion=conflict_canvas.bbox("all"))

    def refresh():
        weights = DYNAMIC_WEIGHTS.copy() if DYNAMIC_WEIGHTS else {}
        if not weights and PERSONA_SPEC:
            for part in PERSONA_SPEC.split("+"):
                p = part.strip()
                name = p.rsplit(":", 1)[0].strip() if ":" in p else p
                if name in PERSONAS:
                    weights[name] = int(p.split(":")[1]) if ":" in p else 50
        w_items = [(k, v) for k, v in sorted(weights.items())]
        draw_animated_bars(weight_canvas, w_items, COLORS, tag="wbars")

        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                ce = [json.loads(line) for line in f if line.strip()]
            fs_by_src = defaultdict(list)
            for e in ce:
                fs_by_src[e.get("source", "unknown")].append(
                    freshness_score(e.get("source", ""), e.get("cached_at", "")))
            c_items = [(src, int(sum(scores) / len(scores))) for src, scores in fs_by_src.items()]
            draw_animated_bars(cache_canvas, c_items, {}, max_val=100, tag="cbars")

        draw_heatmap()
        rebuild_conflicts()

        tq = CACHE_HITS + CACHE_MISSES
        hr = f"{CACHE_HITS / tq * 100:.0f}%" if tq else "N/A"
        ds = len(load_dataset())
        esc_count = len(ESCALATION_EVENTS)
        if os.path.exists(ESCALATION_LOG_PATH):
            with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
                esc_count = sum(1 for _ in f if _.strip())
        status_bar.config(text=f"Dataset: {ds} | Cache: {hr} hit rate | Escalations: {esc_count}")

        root.after(3000, refresh)

    root.after(500, refresh)
    root.mainloop()


def chat_with_feeling(persona_name="mentor"):
    global LAST_REPLY, LAST_PROMPT, LAST_SCORE, RECENT_SCORES, CONSECUTIVE_LOW, DYNAMIC_WEIGHTS, PERSONA_SPEC
    PERSONA_SPEC = persona_name
    DYNAMIC_WEIGHTS = {}
    for part in persona_name.split("+"):
        p = part.strip(); name = p.rsplit(":", 1)[0].strip() if ":" in p else p
        if name in PERSONAS: DYNAMIC_WEIGHTS[name] = 50
    data = load_dataset()
    active_persona = build_persona(data, persona_name)
    print(f"Chat started [persona: {persona_name}]")
    print("Commands: exit | 99 | correct | stats | validate | gold | conflicts | resolve | resolve-skip | dashboard | heatmap | forecast")
    if EXTERNAL_ENABLED: print(f"External: {', '.join(EXTERNAL_SOURCES)}")
    try:
        while True:
            ui = input("You: "); cmd = ui.strip().lower()
            if cmd in ("exit", "quit"): print("Bye."); break
            if cmd == "99": data = load_dataset(); active_persona = build_persona(data, persona_name, DYNAMIC_WEIGHTS); print(f"Reconnected ({avg_conf_str(RECENT_SCORES)})."); continue
            if cmd == "stats": dataset_stats(); continue
            if cmd == "validate": validate_dataset(); continue
            if cmd == "gold": export_gold_dataset(); continue
            if cmd == "conflicts": show_conflicts(); continue
            if cmd == "dashboard": show_dashboard(); continue
            if cmd == "heatmap": show_escalation_heatmap(); continue
            if cmd == "forecast":
                f = forecast_persona()
                if not f: print("   Not enough data to forecast."); continue
                print(f"   Suggested blend: {f}")
                inp = input("   Switch to this blend? (y/n): ").strip().lower()
                if inp == "y":
                    persona_name = f; DYNAMIC_WEIGHTS = {}
                    for part in persona_name.split("+"):
                        p = part.strip(); name = p.rsplit(":", 1)[0].strip() if ":" in p else p
                        if name in PERSONAS: DYNAMIC_WEIGHTS[name] = int(p.split(":")[1]) if ":" in p else 50
                    active_persona = build_persona(data, persona_name, DYNAMIC_WEIGHTS)
                    print(f"   Switched to blend: {persona_name}")
                continue
            if cmd.startswith("resolve ") and len(cmd.split()) >= 2:
                parts = ui.split(maxsplit=2)
                try: idx = int(parts[1]); resolve_conflict(idx, parts[2] if len(parts) > 2 else "")
                except (ValueError, IndexError): print("Usage: resolve <i> <text>")
                continue
            if cmd.startswith("resolve-skip") and len(cmd.split()) >= 2:
                try: skip_conflict(int(cmd.split()[1]))
                except (ValueError, IndexError): print("Usage: resolve-skip <i>")
                continue
            if cmd.startswith("correct") and LAST_REPLY is not None:
                fix = ui[len("correct"):].strip()
                if fix: append_to_dataset(LAST_PROMPT, fix); generate_variations(LAST_PROMPT, fix); LAST_REPLY = fix; print("Corrected.")
                continue
            prompt = apply_template(ui, DYNAMIC_WEIGHTS)
            if DYNAMIC_WEIGHTS and "+" in persona_name:
                active_persona = build_persona(data, persona_name, DYNAMIC_WEIGHTS)
                print(f"   Weights: {DYNAMIC_WEIGHTS}")
                log_weights(DYNAMIC_WEIGHTS)
            try:
                resp = ollama.chat(model=MODEL_VERSION, messages=[
                    {"role": "system", "content": active_persona},
                    {"role": "user", "content": prompt},
                ])
                reply = resp["message"]["content"]; score = score_reply(reply)
                RECENT_SCORES.append(score)
                if len(RECENT_SCORES) > 20: RECENT_SCORES.pop(0)
                CONSECUTIVE_LOW = CONSECUTIVE_LOW + 1 if score < 50 else 0
                print(f"Ollama: {reply}  [{score}/100]")
                LAST_PROMPT, LAST_REPLY, LAST_SCORE = prompt, reply, score
                if score < 50:
                    append_to_weak(prompt, reply, score)
                    tr = handle_escalation_tier(CONSECUTIVE_LOW, ui, prompt)
                    if tr: print("Escalation:", tr[:100]); append_to_dataset(prompt, tr)
                    else: print("Fallback:", FALLBACK); append_to_dataset(prompt, FALLBACK)
                elif score < SCORE_THRESHOLD:
                    append_to_weak(prompt, reply, score)
                    fix = input("Correct? (blank to skip): ").strip()
                    if fix: append_to_dataset(prompt, fix); generate_variations(prompt, fix)
                else:
                    if score > 90: print("Auto-approved.")
                    append_to_dataset(prompt, reply)
            except Exception: print("Fallback:", FALLBACK)
    except KeyboardInterrupt: print("\nInterrupted.")


def avg_conf_str(s):
    return f"avg {sum(s)/len(s):.0f}/100" if s else "no data"


def init_voice():
    global VOICE_RECOGNIZER, VOICE_ENGINE, VOICE_ENABLED
    try:
        VOICE_RECOGNIZER = sr.Recognizer()
        VOICE_RECOGNIZER.energy_threshold = 3000
        VOICE_RECOGNIZER.dynamic_energy_threshold = True
        VOICE_RECOGNIZER.pause_threshold = 0.8
        VOICE_ENGINE = pyttsx3.init()
        VOICE_ENGINE.setProperty("rate", 160)
        VOICE_ENGINE.setProperty("volume", 0.9)
        voices = VOICE_ENGINE.getProperty("voices")
        if voices:
            for v in voices:
                if "zira" in v.name.lower() or "david" in v.name.lower():
                    VOICE_ENGINE.setProperty("voice", v.id)
                    break
        VOICE_ENABLED = True
        return True
    except Exception as e:
        print(f"   Voice init failed: {e}")
        return False


def apply_persona_voice(persona_name):
    if not VOICE_ENGINE:
        return
    parts = persona_name.split("+")
    primary = parts[0].split(":")[0].strip()
    vc = VOICE_PERSONA_VOICES.get(primary, {"rate": 160, "volume": 0.9})
    VOICE_ENGINE.setProperty("rate", vc["rate"])
    VOICE_ENGINE.setProperty("volume", vc["volume"])
    target = vc.get("voice_name", "").lower()
    if target:
        voices = VOICE_ENGINE.getProperty("voices")
        for v in voices:
            if target in v.name.lower():
                VOICE_ENGINE.setProperty("voice", v.id)
                break


ADAPTIVE_CUES = {
    "motivate": [(523, 150), (659, 150), (784, 200)],
    "explain": [(440, 200), (440, 200), (440, 200)],
    "advise": [(784, 150), (659, 150), (523, 200)],
}

def weighted_voice_schedule(weights, total):
    if not weights or total <= 0:
        return []
    w_sum = sum(weights.values())
    if w_sum <= 0:
        return []
    schedule = []
    for persona, weight in weights.items():
        count = max(1, round(weight / w_sum * total))
        schedule.extend([persona] * count)
    import random
    random.shuffle(schedule)
    return schedule[:total]


def play_escalation_cue(tier, intent=None):
    notes = ADAPTIVE_CUES.get(intent)
    if notes:
        freqs = [n[0] for n in notes]
        durs = [n[1] for n in notes]
        for i in range(min(tier, len(freqs))):
            winsound.Beep(freqs[i], durs[i])
            if i < tier - 1 and i < len(freqs) - 1:
                time.sleep(0.06)
    else:
        if tier == 1:
            winsound.Beep(400, 200)
        elif tier == 2:
            winsound.Beep(500, 150)
            time.sleep(0.1)
            winsound.Beep(500, 150)
        elif tier == 3:
            winsound.Beep(700, 100)
            time.sleep(0.08)
            winsound.Beep(700, 100)
            time.sleep(0.08)
            winsound.Beep(700, 100)


def speak_text(text):
    if not VOICE_ENGINE:
        return
    def _speak():
        VOICE_ENGINE.say(text)
        VOICE_ENGINE.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()


def speak_dashboard_summary():
    data = load_dataset()
    parts = []
    if data:
        parts.append(f"Dataset has {len(data)} entries.")
    if RECENT_SCORES:
        avg = sum(RECENT_SCORES) / len(RECENT_SCORES)
        parts.append(f"Average confidence is {avg:.0f} percent.")
    if os.path.exists(ESCALATION_LOG_PATH):
        with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        if events:
            tiers = Counter(e.get("tier") for e in events)
            parts.append(f"{len(events)} escalation events logged.")
            for t in sorted(tiers):
                label = {1: "external lookups", 2: "persona shifts", 3: "hard fallbacks"}.get(t, f"tier {t}")
                parts.append(f"{tiers[t]} {label}.")
    if os.path.exists(WEIGHT_LOG_PATH):
        with open(WEIGHT_LOG_PATH, encoding="utf-8") as f:
            history = [json.loads(line) for line in f if line.strip()]
        if history:
            weights = history[-1].get("weights", {})
            w_strs = [f"{k} {v}" for k, v in sorted(weights.items())]
            if w_strs:
                parts.append("Current weights: " + ", ".join(w_strs) + ".")
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache_entries = [json.loads(line) for line in f if line.strip()]
        parts.append(f"Cache has {len(cache_entries)} entries.")
    summary = " ".join(parts)
    print(f"   Voice dashboard: {summary}")
    speak_text(summary)


def listen_speech(timeout=5, phrase_limit=10):
    if not VOICE_RECOGNIZER:
        return None
    try:
        with sr.Microphone() as source:
            print("   (listening...)")
            VOICE_RECOGNIZER.adjust_for_ambient_noise(source, duration=0.3)
            try:
                audio = VOICE_RECOGNIZER.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            except sr.WaitTimeoutError:
                print("   (no speech detected)")
                return None
        print("   (processing...)")
        text = VOICE_RECOGNIZER.recognize_google(audio)
        print(f"   (heard: {text})")
        return text
    except sr.UnknownValueError:
        print("   (could not understand audio)")
        return None
    except sr.RequestError as e:
        print(f"   (speech service error: {e})")
        return None
    except Exception as e:
        print(f"   (mic error: {e})")
        return None


def chat_with_voice(persona_name="mentor"):
    global LAST_REPLY, LAST_PROMPT, LAST_SCORE, RECENT_SCORES, CONSECUTIVE_LOW, DYNAMIC_WEIGHTS, PERSONA_SPEC
    if not init_voice():
        print("Voice init failed, falling back to text mode.")
        chat_with_feeling(persona_name)
        return
    print("Voice chat active. Speak into your microphone.")
    print("Say 'exit' or 'quit' to stop. Say 'command' to see available commands.")
    apply_persona_voice(persona_name)
    PERSONA_SPEC = persona_name
    DYNAMIC_WEIGHTS = {}
    for part in persona_name.split("+"):
        p = part.strip()
        name = p.rsplit(":", 1)[0].strip() if ":" in p else p
        if name in PERSONAS:
            DYNAMIC_WEIGHTS[name] = 50
    data = load_dataset()
    active_persona = build_persona(data, persona_name)
    if EXTERNAL_ENABLED:
        print(f"External sources: {', '.join(EXTERNAL_SOURCES)}")
    try:
        while True:
            ui = listen_speech()
            if ui is None:
                continue
            cmd = ui.strip().lower()
            if cmd in ("exit", "quit"):
                speak_text("Goodbye.")
                print("Bye.")
                break
            if cmd in ("command", "commands"):
                cmds = "Commands: exit, stats, validate, gold, conflicts, dashboard, heatmap, forecast, correct, resolve"
                print(cmds)
                speak_text("Available commands: stats, validate, gold, dashboard, heatmap, forecast, correct, and resolve")
                continue
            if cmd == "stats":
                dataset_stats()
                speak_text("Stats displayed on screen.")
                continue
            if cmd == "validate":
                validate_dataset()
                continue
            if cmd == "gold":
                export_gold_dataset()
                continue
            if cmd == "conflicts":
                show_conflicts()
                continue
            if cmd == "dashboard":
                show_dashboard()
                speak_dashboard_summary()
                continue
            if cmd == "heatmap":
                show_escalation_heatmap()
                continue
            if cmd == "forecast":
                f = forecast_persona()
                if not f:
                    speak_text("Not enough data to forecast persona.")
                    continue
                speak_text(f"Suggested blend: {f}")
                apply_persona_voice(f)
                continue
            if cmd.startswith("resolve "):
                parts = ui.split(maxsplit=2)
                try:
                    idx = int(parts[1])
                    resolve_conflict(idx, parts[2] if len(parts) > 2 else "", method="voice")
                except (ValueError, IndexError):
                    speak_text("Usage: resolve number text")
                continue
            if cmd.startswith("correct") and LAST_REPLY is not None:
                fix = ui[len("correct"):].strip()
                if fix:
                    append_to_dataset(LAST_PROMPT, fix)
                    generate_variations(LAST_PROMPT, fix)
                    LAST_REPLY = fix
                    print("Corrected.")
                continue
            if cmd in ("train model", "train"):
                speak_text("Training model from dataset.")
                train_from_dataset()
                continue
            if any(w in cmd for w in ["that was weak", "bad reply", "weak answer"]):
                if LAST_PROMPT and LAST_REPLY:
                    append_to_weak(LAST_PROMPT, LAST_REPLY, LAST_SCORE or 0)
                    speak_text("Last reply logged as weak.")
                else:
                    speak_text("No reply to flag.")
                continue
            if cmd == "retrain":
                if os.path.exists(WEAK_REPLIES_PATH):
                    with open(WEAK_REPLIES_PATH, encoding="utf-8") as f:
                        weak_count = sum(1 for _ in f if _.strip())
                    speak_text(f"Retraining with {weak_count} weak examples.")
                    train_from_dataset()
                    speak_training_narration()
                else:
                    speak_text("No weak replies to retrain on.")
                continue
            if cmd == "resolve conflicts":
                resolved = auto_resolve_conflicts()
                if resolved:
                    speak_text(f"Auto-resolved {resolved} conflicts.")
                else:
                    show_conflicts()
                    speak_text("Showing unresolved conflicts.")
                continue
            if cmd in ("show heatmap", "heatmap"):
                show_escalation_heatmap()
                speak_text("Heatmap displayed on screen.")
                continue
            if cmd in ("export gold", "gold"):
                export_gold_dataset()
                speak_text("Gold dataset exported.")
                continue
            if cmd in ("narrate training", "training narration", "training summary"):
                speak_training_narration()
                continue
            if cmd in ("narrate contributors", "contributor summary", "contributor narration"):
                speak_text(f"Contributor analytics. {USER_ID}: {len(TRAINING_EVENTS)} training events tracked.")
                continue
            if cmd in ("show cache", "cache stats"):
                show_cache_stats()
                speak_text("Cache stats on screen.")
                continue
            if cmd in ("sync", "sync data"):
                sync_dataset(user_id=USER_ID)
                speak_text(f"Sync complete for {USER_ID}.")
                continue
            if cmd in ("show stats", "stats"):
                dataset_stats()
                speak_text("Stats on screen.")
                continue
            prompt = apply_template(ui, DYNAMIC_WEIGHTS)
            if DYNAMIC_WEIGHTS and "+" in persona_name:
                active_persona = build_persona(data, persona_name, DYNAMIC_WEIGHTS)
                print(f"   Weights: {DYNAMIC_WEIGHTS}")
                log_weights(DYNAMIC_WEIGHTS)
            try:
                reply_chunks = []
                tts_buffer = ""
                speak_queue = queue.Queue()
                tts_done = threading.Event()
                voice_schedule = []
                voice_idx = 0

                if "+" in persona_name and DYNAMIC_WEIGHTS:
                    blended = {k: v for k, v in DYNAMIC_WEIGHTS.items() if k in PERSONAS}
                    if len(blended) > 1:
                        voice_schedule = weighted_voice_schedule(blended, 30)

                def tts_worker():
                    nonlocal voice_idx
                    while True:
                        item = speak_queue.get()
                        if item is None:
                            break
                        if isinstance(item, tuple):
                            text, vp = item
                        else:
                            text, vp = item, None
                        if vp and voice_schedule:
                            vc = VOICE_PERSONA_VOICES.get(vp, {})
                            target = vc.get("voice_name", "").lower()
                            if target:
                                voices = VOICE_ENGINE.getProperty("voices")
                                for v in voices:
                                    if target in v.name.lower():
                                        VOICE_ENGINE.setProperty("voice", v.id)
                                        break
                            rate = vc.get("rate", 160)
                            vol = vc.get("volume", 0.9)
                            VOICE_ENGINE.setProperty("rate", rate)
                            VOICE_ENGINE.setProperty("volume", vol)
                        VOICE_ENGINE.say(text)
                        VOICE_ENGINE.runAndWait()
                    tts_done.set()

                tts_thread = threading.Thread(target=tts_worker, daemon=True)
                tts_thread.start()

                stream = ollama.chat(model=MODEL_VERSION, messages=[
                    {"role": "system", "content": active_persona},
                    {"role": "user", "content": prompt},
                ], stream=True)
                print("Ollama: ", end="", flush=True)
                for chunk in stream:
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        reply_chunks.append(content)
                        print(content, end="", flush=True)
                        tts_buffer += content
                        sentences = re.split(r'(?<=[.!?])\s+', tts_buffer)
                        tts_buffer = ""
                        for i, s in enumerate(sentences):
                            stripped = s.strip()
                            if not stripped:
                                continue
                            if i == len(sentences) - 1 and not stripped[-1] in ".!?":
                                tts_buffer = stripped
                            else:
                                vp = voice_schedule[voice_idx % len(voice_schedule)] if voice_schedule else None
                                voice_idx += 1
                                speak_queue.put((stripped, vp))
                print()
                if tts_buffer.strip():
                    vp = voice_schedule[voice_idx % len(voice_schedule)] if voice_schedule else None
                    speak_queue.put((tts_buffer.strip(), vp))
                speak_queue.put(None)
                tts_done.wait(timeout=30)
                reply = "".join(reply_chunks)
                score = score_reply(reply)
                RECENT_SCORES.append(score)
                if len(RECENT_SCORES) > 20:
                    RECENT_SCORES.pop(0)
                CONSECUTIVE_LOW = CONSECUTIVE_LOW + 1 if score < 50 else 0
                print(f"  [{score}/100]")
                LAST_PROMPT, LAST_REPLY, LAST_SCORE = prompt, reply, score
                TRAINING_EVENTS.append({
                    "type": "weak_reply" if score < SCORE_THRESHOLD else "good_reply",
                    "score": score,
                    "intent": detect_intent(ui) or "general",
                    "timestamp": datetime.now().isoformat(),
                })
                if score < 50:
                    append_to_weak(prompt, reply, score)
                    esc_intent = detect_intent(ui)
                    if CONSECUTIVE_LOW >= 7:
                        play_escalation_cue(3, esc_intent)
                    elif CONSECUTIVE_LOW >= 5:
                        play_escalation_cue(2, esc_intent)
                    elif CONSECUTIVE_LOW >= 3:
                        play_escalation_cue(1, esc_intent)
                    tr = handle_escalation_tier(CONSECUTIVE_LOW, ui, prompt)
                    if tr:
                        print("Escalation:", tr[:100])
                        append_to_dataset(prompt, tr)
                        speak_text(tr)
                    else:
                        print("Fallback:", FALLBACK)
                        append_to_dataset(prompt, FALLBACK)
                        speak_text(FALLBACK)
                elif score < SCORE_THRESHOLD:
                    append_to_weak(prompt, reply, score)
                    print("   (low confidence - reply logged for review)")
                else:
                    if score > 90:
                        print("Auto-approved.")
                    append_to_dataset(prompt, reply)
            except Exception as e:
                print(f"Fallback: {FALLBACK}")
                speak_text(FALLBACK)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        speak_text("Interrupted.")


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


def simulate_persona(rounds=10, blends=None):
    if blends is None:
        blends = [
            {"name": "Mentor only", "persona": "mentor", "weights": {"mentor": 100, "coach": 0, "teacher": 0}},
            {"name": "Coach only", "persona": "coach", "weights": {"mentor": 0, "coach": 100, "teacher": 0}},
            {"name": "Teacher only", "persona": "teacher", "weights": {"mentor": 0, "coach": 0, "teacher": 100}},
            {"name": "Balanced", "persona": "mentor:34+coach:33+teacher:33", "weights": {"mentor": 34, "coach": 33, "teacher": 33}},
            {"name": "Mentor-heavy", "persona": "mentor:60+coach:20+teacher:20", "weights": {"mentor": 60, "coach": 20, "teacher": 20}},
        ]
    results = []
    for blend in blends:
        scores = []
        escalations = 0
        weaks = 0
        intents = Counter()
        dy_weights = dict(blend.get("weights", {}))
        for i in range(rounds):
            prompt = SIMULATION_PROMPTS[i % len(SIMULATION_PROMPTS)]
            personal = build_persona(blend["persona"], dy_weights)
            reply = generate_mock_reply(prompt, blend["persona"])
            score = score_reply(reply)
            scores.append(score)
            intent = detect_intent(prompt)
            if intent:
                intents[intent] += 1
            if score < 50:
                weaks += 1
            if score < 50 and len(scores) >= 3 and all(s < 50 for s in scores[-3:]):
                escalations += 1
        avg_c = round(sum(scores) / len(scores), 1) if scores else 0
        results.append({
            "name": blend["name"],
            "avg_confidence": avg_c,
            "escalations": escalations,
            "weak_ratio": f"{round(weaks / rounds * 100, 1)}%",
            "weak_count": weaks,
            "best_intent": intents.most_common(1)[0][0] if intents else "none",
        })
    sep = "-" * 72
    print()
    print("  Persona Simulation Results")
    print(sep)
    print(f"  {'Blend':<22} {'Avg Conf':>9} {'Escal.':>7} {'Weak%':>7} {'Weak#':>6} {'Top Intent':<14}")
    print(sep)
    for r in results:
        print(f"  {r['name']:<22} {r['avg_confidence']:>8.1f}% {r['escalations']:>7} {r['weak_ratio']:>7} {r['weak_count']:>6} {r['best_intent']:<14}")
    print(sep)
    print(f"  Prompts per blend: {rounds}  |  Total prompts: {rounds * len(blends)}")
    print()
    return results


ADAPTIVE_REINFORCED = set()
ESCALATION_REINFORCED_CLI = set()
CLI_FEWSHOT_TEMPLATES = {
    "motivate": {"prompt": "I'm feeling really down and need motivation to keep going.", "completion": "I hear you. It's completely okay to have days where you feel low. What matters is that you're still here. Let's take one small step together."},
    "explain": {"prompt": "Can you explain this to me like I'm five years old?", "completion": "Of course! Let me start with the simplest way to think about it. The core idea is really just one small concept, and once that clicks, everything else builds on it naturally."},
    "advise": {"prompt": "I'm stuck between two choices and don't know what to do.", "completion": "That feeling of being stuck is common. Let's break this down. What does your gut tell you? Which option aligns with your long-term values? You don't need the perfect answer."},
    "general": {"prompt": "I don't know what to do with my life.", "completion": "That's a deeply honest question. You don't need to have it all figured out today. What if we focused on just the next season? What would feel meaningful to explore?"},
}


CLI_ESCALATION_TEMPLATES = {
    "motivate": {"prompt": "I keep failing and I'm losing motivation completely.", "completion": "Failure is not the opposite of success — it's part of it. Let's look at what you've learned. What's one small thing you could try differently?"},
    "explain": {"prompt": "I've read this three times and I still don't understand.", "completion": "Let's try a completely different approach. Forget what you've read — let me show you with a real-world example you already understand."},
    "advise": {"prompt": "Every option I consider seems terrible. I'm completely stuck.", "completion": "When every path feels wrong, pick the option that feels the least harmful. Sometimes unblocking yourself matters more than finding the perfect answer."},
    "general": {"prompt": "Nothing is working and I don't know what to do anymore.", "completion": "Stop trying to solve everything at once. Pick one tiny thing you can control right now and do only that. Progress is rebuilt one small win at a time."},
}


def escalation_reinforce_cli():
    if not os.path.exists(ESCALATION_LOG_PATH):
        print("No escalation log found.")
        return
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events:
        print("No escalation events.")
        return
    patterns = Counter((e.get("intent", "general"), e.get("tier", 1)) for e in events)
    added = 0
    for (intent, tier), count in patterns.most_common():
        key = f"{intent}_tier{tier}"
        if key in ESCALATION_REINFORCED_CLI or count < 2:
            continue
        ESCALATION_REINFORCED_CLI.add(key)
        t = CLI_ESCALATION_TEMPLATES.get(intent, CLI_ESCALATION_TEMPLATES["general"])
        entry = {"intent": intent, "tier": tier, "escalation_count": count, "prompt": t["prompt"],
                 "completion": t["completion"], "generated_at": datetime.now().isoformat()}
        with open("fewshot_dataset.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"   Escalation-reinforced '{intent}' tier {tier} ({count} events)")
        added += 1
    if not added:
        print("   No escalation patterns to reinforce.")
    return added


def adaptive_reinforce(rounds=8, threshold=30):
    intents_weak = defaultdict(int)
    for i in range(rounds):
        prompt = SIMULATION_PROMPTS[i % len(SIMULATION_PROMPTS)]
        reply = generate_mock_reply(prompt, "mentor")
        score = score_reply(reply)
        if score < 50:
            intent = detect_intent(prompt) or "general"
            intents_weak[intent] += 1
    added = []
    for intent, count in sorted(intents_weak.items(), key=lambda x: -x[1]):
        pct = (count / rounds) * 100
        if pct >= threshold and intent not in ADAPTIVE_REINFORCED:
            ADAPTIVE_REINFORCED.add(intent)
            t = CLI_FEWSHOT_TEMPLATES.get(intent, CLI_FEWSHOT_TEMPLATES["general"])
            entry = {"intent": intent, "prompt": t["prompt"], "completion": t["completion"],
                     "generated_at": datetime.now().isoformat(), "weak_rate": round(pct, 1)}
            with open("fewshot_dataset.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"   Auto-reinforced '{intent}' (weak rate {pct:.1f}%)")
            added.append(intent)
    if not added:
        print("   All intents already reinforced or below threshold.")
    return added


if __name__ == "__main__":
    persona_name, sync_target, merge_policy, voice_enabled, gui_enabled, narrate_enabled = "mentor", None, "highest", False, False, False
    global USER_ID
    narrate_interval = 5
    rest = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--persona" and i + 1 < len(sys.argv): persona_name = sys.argv[i + 1]; i += 2
        elif a in ("--sync", "--sync-dir") and i + 1 < len(sys.argv): sync_target = sys.argv[i + 1]; i += 2
        elif a == "--merge-policy" and i + 1 < len(sys.argv): merge_policy = sys.argv[i + 1]; i += 2
        elif a == "--conflict-policy" and i + 1 < len(sys.argv):
            v = sys.argv[i + 1].lower()
            if v in ("low", "medium", "all"): CONFLICT_AUTO_POLICY = v
            i += 2
        elif a == "--user" and i + 1 < len(sys.argv): USER_ID = sys.argv[i + 1]; i += 2
        elif a == "--voice": voice_enabled = True; i += 1
        elif a == "--gui": gui_enabled = True; i += 1
        elif a == "--narrate": narrate_enabled = True; i += 1
        elif a == "--narrate-interval" and i + 1 < len(sys.argv): narrate_interval = int(sys.argv[i + 1]); narrate_enabled = True; i += 2
        elif a == "--auto-refresh": AUTO_REFRESH_ENABLED = True; i += 1
        elif a == "--federated-sync" and i + 1 < len(sys.argv): print(f"Syncing as {USER_ID}"); federated_sync(sys.argv[i + 1], merge_policy); sys.exit(0)
        elif a == "--conflicts": show_conflicts(); sys.exit(0)
        elif a == "--dashboard": show_dashboard(); sys.exit(0)
        elif a == "--heatmap": show_escalation_heatmap(); sys.exit(0)
        elif a == "--forecast":
            f = forecast_persona()
            if f: print(f"Suggested blend: {f}")
            else: print("Not enough data.")
            sys.exit(0)
        elif a == "--adaptive-reinforce":
            rounds = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit() else 8
            if rounds != 8: i += 1
            adaptive_reinforce(rounds=rounds); sys.exit(0)
        elif a == "--federated-simulate":
            rounds = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit() else 8
            if rounds != 8: i += 1
            simulate_persona(rounds=rounds); sys.exit(0)
        elif a == "--simulate":
            rounds = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit() else 10
            if rounds != 10: i += 1
            simulate_persona(rounds=rounds); sys.exit(0)
        elif a == "--reinforce": reinforce_hotspots(); sys.exit(0)
        elif a == "--escalation-reinforce":
            escalation_reinforce_cli(); sys.exit(0)
        else: rest.append(a); i += 1
    sys.argv = [sys.argv[0]] + rest
    if narrate_enabled and voice_enabled:
        init_voice()
        if VOICE_ENGINE:
            start_narration_scheduler(narrate_interval)
            print(f"   Narration scheduled every {narrate_interval} min(s).")
    if gui_enabled and not voice_enabled:
        import threading as _thr
        _thr.Thread(target=dashboard_gui, daemon=True).start()
        chat_with_feeling(persona_name=persona_name)
    elif gui_enabled and voice_enabled:
        import threading as _thr
        _thr.Thread(target=dashboard_gui, daemon=True).start()
        chat_with_voice(persona_name=persona_name)
    elif "--stats" in sys.argv: dataset_stats()
    elif "--validate" in sys.argv: validate_dataset()
    elif "--gold" in sys.argv: export_gold_dataset()
    elif "--train" in sys.argv: train_from_dataset()
    elif "--sync" in sys.argv or sync_target: sync_dataset(sync_target)
    elif voice_enabled: chat_with_voice(persona_name=persona_name)
    else: chat_with_feeling(persona_name=persona_name)
