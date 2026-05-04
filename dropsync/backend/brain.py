import httpx
import json

# --- Ollama: fast, local, real-time decisions ---
def pick_next_track(
    current_track: dict,
    queue: list[dict],
    vibe_mode: str  # "sangeet" | "club" | "lounge" | "baaraat"
) -> dict:
    """
    Call local Ollama (Mistral) with current track metadata + queue.
    """
    prompt = f"""
    You are an AI DJ.
    Current track: {json.dumps(current_track)}
    Vibe mode: {vibe_mode}
    Queue: {json.dumps(queue)}
    
    Pick the best next track from the queue based on BPM, key, and vibe.
    Return ONLY the exact JSON object of the selected track from the queue.
    """
    
    try:
        response = httpx.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        result_text = response.json()["response"]
        return json.loads(result_text)
    except Exception as e:
        if queue:
            return queue[0]
        return {}

# --- Local Ollama fallback for set planning ---
def plan_set_local(tracks, vibe_mode, duration_minutes, event_context):
    """
    Local Ollama alternative to Claude for planning the set arc.
    """
    prompt = f"""
    You are an AI DJ. Plan a {duration_minutes} minute set for: {event_context}
    Vibe mode: {vibe_mode}
    
    Tracks available (JSON):
    {json.dumps(tracks)}
    
    Return ONLY valid JSON with keys:
    set_arc, drop_script, singalong_tracks, total_duration_estimate
    """
    
    response = httpx.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False
    })
    
    try:
        result_text = response.json()["response"]
        # Basic cleanup in case Ollama wraps the JSON in markdown blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
            
        return json.loads(result_text.strip())
    except Exception as e:
        return {"error": str(e), "raw_response": response.json().get("response", "")}
