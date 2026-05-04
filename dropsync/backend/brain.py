import httpx
import json
import os
from typing import Union
from dj_algo import score_next_track

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ─────────────────────────────────────────
# HELPER: Check if Ollama is running
# ─────────────────────────────────────────
def ollama_is_available() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────
# HELPER: Clean JSON from Ollama response
# ─────────────────────────────────────────
def extract_json(text: str) -> Union[dict, list]:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    # Find first { or [
    start = min(
        text.find('{') if '{' in text else len(text),
        text.find('[') if '[' in text else len(text)
    )
    end_brace = text.rfind('}')
    end_bracket = text.rfind(']')
    end = max(end_brace, end_bracket) + 1
    return json.loads(text[start:end].strip())


# ─────────────────────────────────────────
# 1. PICK NEXT TRACK — Ollama real-time
# ─────────────────────────────────────────
def pick_next_track(
    current_track: dict,
    queue: list[dict],
    vibe_mode: str,
    position_in_set: float = 0.5
) -> dict:
    """
    Fast local decision — Mistral picks next track from queue.
    Falls back to score_next_track() if Ollama is unavailable or returns bad JSON.
    """
    if not queue:
        return {}

    # Always score tracks locally as a baseline
    scored = sorted(
        queue,
        key=lambda t: score_next_track(current_track, t, vibe_mode, position_in_set),
        reverse=True
    )

    # Try Ollama for a smarter pick
    if not ollama_is_available():
        return scored[0]  # Pure algo fallback

    prompt = f"""You are an AI DJ. Pick the single best next track.

Current track:
- BPM: {current_track.get('bpm')}
- Key: {current_track.get('key')}
- Mood: {current_track.get('mood')}
- Energy: {round(sum(current_track.get('energy_curve', [0.5])) / max(len(current_track.get('energy_curve', [1])), 1), 2)}

Vibe mode: {vibe_mode}
Position in set: {position_in_set} (0=start, 1=end)

Queue (pick one):
{json.dumps([{"title": t.get("title"), "bpm": t.get("bpm"), "key": t.get("key"), "mood": t.get("mood")} for t in queue], indent=2)}

Return ONLY the title of the best next track as a JSON string: {{"title": "..."}}"""

    try:
        response = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=30.0
        )
        result = extract_json(response.json()["response"])
        chosen_title = result.get("title", "")

        # Find matching track in queue
        match = next((t for t in queue if t.get("title") == chosen_title), None)
        return match if match else scored[0]

    except Exception:
        return scored[0]  # Algo fallback if Ollama fails


# ─────────────────────────────────────────
# 2. PLAN SET — Claude API (premium)
# ─────────────────────────────────────────
def plan_set(
    tracks: list[dict],
    vibe_mode: str,
    duration_minutes: int,
    event_context: str
) -> dict:
    """
    Full set planning via Claude API.
    Falls back to plan_set_local() if API key missing or call fails.
    """
    if not ANTHROPIC_API_KEY:
        return plan_set_local(tracks, vibe_mode, duration_minutes, event_context)

    is_indian_event = vibe_mode in ["sangeet", "baaraat"]

    prompt = f"""You are an expert AI DJ planning a live set for an Indian event.

Event: {event_context}
Vibe mode: {vibe_mode}
Target duration: {duration_minutes} minutes
{"Use Hindi-English mix in drop_script." if is_indian_event else ""}

Available tracks:
{json.dumps([
    {
        "index": i,
        "title": t.get("title", f"Track {i}"),
        "bpm": t.get("bpm"),
        "key": t.get("key"),
        "mood": t.get("mood"),
        "danceability": t.get("danceability"),
        "duration": t.get("duration")
    }
    for i, t in enumerate(tracks)
], indent=2)}

Return ONLY valid JSON:
{{
  "set_arc": [
    {{"track_index": 0, "position": "opener", "notes": "why this track here"}},
    {{"track_index": 2, "position": "build", "notes": "..."}},
    {{"track_index": 4, "position": "peak", "notes": "main drop moment"}},
    {{"track_index": 1, "position": "cooldown", "notes": "..."}}
  ],
  "drop_script": "crowd callout line here",
  "singalong_tracks": [0, 4],
  "total_duration_estimate": 58
}}"""

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30.0
        )
        text = response.json()["content"][0]["text"]
        return extract_json(text)

    except Exception as e:
        return plan_set_local(tracks, vibe_mode, duration_minutes, event_context)


# ─────────────────────────────────────────
# 3. PLAN SET — Local Ollama fallback
# ─────────────────────────────────────────
def plan_set_local(
    tracks: list[dict],
    vibe_mode: str,
    duration_minutes: int,
    event_context: str
) -> dict:
    """
    Full set planning via local llama3.1.
    Pure algo fallback if Ollama also unavailable.
    """
    if not ollama_is_available():
        # Pure deterministic fallback — sort by danceability
        sorted_tracks = sorted(
            enumerate(tracks),
            key=lambda x: x[1].get("danceability", 0.5)
        )
        arc_positions = ["opener", "build", "peak", "cooldown"]
        set_arc = [
            {
                "track_index": idx,
                "position": arc_positions[min(i, len(arc_positions)-1)],
                "notes": "Auto-sorted by danceability"
            }
            for i, (idx, _) in enumerate(sorted_tracks)
        ]
        return {
            "set_arc": set_arc,
            "drop_script": "Ab sab ready ho jao!",
            "singalong_tracks": [],
            "total_duration_estimate": duration_minutes
        }

    prompt = f"""You are an AI DJ. Plan a {duration_minutes} minute set.
Event: {event_context}
Vibe: {vibe_mode}

Tracks:
{json.dumps([{"index": i, "title": t.get("title", f"Track {i}"), "bpm": t.get("bpm"), "mood": t.get("mood")} for i, t in enumerate(tracks)])}

Return ONLY valid JSON with keys: set_arc (list of {{track_index, position, notes}}), drop_script (string), singalong_tracks (list of ints), total_duration_estimate (int)."""

    try:
        response = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": "llama3.1", "prompt": prompt, "stream": False},
            timeout=60.0
        )
        return extract_json(response.json()["response"])
    except Exception as e:
        return {"error": str(e)}
