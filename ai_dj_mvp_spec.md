# AI DJ MVP — Antigravity Build Spec

## Project Name
**DropSync** — An AI-powered DJ engine that fetches audio from YouTube, analyzes beats in real time, and mixes tracks like a live DJ using Claude + Ollama for set intelligence.

---

## What We Are Building

A local web app (runs on localhost) that acts as an AI DJ. The user types a vibe or song name, and the system:
1. Finds the track on YouTube via yt-dlp
2. Analyzes it for BPM, beat grid, energy, mood using Librosa + Essentia
3. Uses Ollama (local) for lightweight next-track decisions
4. Uses Claude API for set arc planning and drop scripting
5. Plays and mixes tracks in the browser using Web Audio API
6. Fires ElevenLabs voice drops at key moments

No files are stored. Everything streams. Users never upload anything.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Backend | Python + FastAPI | Audio processing, yt-dlp, Librosa |
| Audio fetch | yt-dlp | YouTube stream URL, no download |
| Audio analysis | Librosa + Essentia | BPM, beats, energy, mood, key |
| Stem separation | Spleeter | Vocal stripping for sing-along mode |
| Local AI | Ollama (Mistral 7B) | Real-time queue decisions, low latency |
| Cloud AI | Claude claude-sonnet-4-20250514 via Anthropic API | Set arc, drop scripts, vibe intelligence |
| Voice drops | ElevenLabs API | DJ shoutouts, crowd callouts |
| Frontend | React + Web Audio API + wavesurfer.js | Deck UI, waveform, EQ, crossfade |
| Communication | WebSocket | Real-time beat events backend → frontend |

---

## Folder Structure to Create

```
dropsync/
├── backend/
│   ├── main.py                  # FastAPI app entry
│   ├── fetcher.py               # yt-dlp YouTube audio stream
│   ├── analyzer.py              # Librosa + Essentia analysis
│   ├── dj_algo.py               # Beat matching, phase align, phrase detect
│   ├── brain.py                 # Ollama + Claude API calls
│   ├── drops.py                 # ElevenLabs voice drop generator
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── DeckUI.jsx       # Main DJ deck
│   │   │   ├── Waveform.jsx     # wavesurfer.js waveform
│   │   │   ├── Queue.jsx        # Upcoming track queue
│   │   │   ├── VibePicker.jsx   # Sangeet / Club / Lounge / Baaraat
│   │   │   ├── EQSliders.jsx    # Bass / Mid / Treble crossfade visual
│   │   │   └── DropAlert.jsx    # "Drop incoming" countdown
│   │   └── hooks/
│   │       ├── useAudioEngine.js  # Web Audio API wrapper
│   │       └── useWebSocket.js    # Beat event listener
│   └── package.json
└── README.md
```

---

## Phase 1 — Backend (Build This First)

### Task 1: `fetcher.py`

Build a function that accepts a search query string and returns a temporary streamable audio URL from YouTube. No file should be downloaded or stored on disk.

```python
# Expected interface
def get_stream_url(query: str) -> dict:
    # Returns:
    # {
    #   "url": "https://...",        # streamable audio URL (expires ~6hrs)
    #   "title": "Song Name",
    #   "duration": 243,             # seconds
    #   "thumbnail": "https://..."
    # }
```

Use `yt-dlp` with `format: bestaudio`. Extract the first result from `ytsearch:{query}`. Do not download. Return the direct stream URL only.

---

### Task 2: `analyzer.py`

Build a function that accepts an audio stream URL and returns a full audio analysis object. This is the most critical module — get it right.

```python
# Expected interface
def analyze_track(stream_url: str) -> dict:
    # Returns:
    # {
    #   "bpm": 128.0,
    #   "bpm_confidence": 0.92,         # 0-1, flag if < 0.7
    #   "key": "F# minor",
    #   "beats": [0.47, 0.94, 1.41...], # beat timestamps in seconds
    #   "downbeats": [0.47, 2.35...],   # beat 1 of each bar (every 4 beats)
    #   "phrase_boundaries": [8.2, 24.6, 48.0...], # energy dip moments
    #   "energy_curve": [0.3, 0.4, 0.8...],        # per-second energy 0-1
    #   "mood": "energetic",            # energetic / calm / aggressive / melancholic
    #   "danceability": 0.78,           # 0-1
    #   "drop_candidates": [45.2, 92.8] # timestamps where drops are likely
    # }
```

**Important — BPM harmonic correction:**
Librosa frequently returns half or double the true BPM. Add a correction layer:
- If detected BPM < 80: multiply by 2
- If detected BPM > 180: divide by 2
- Target range: 80–175 BPM

**Phrase boundary detection logic:**
A phrase boundary is where a new musical section starts — typically every 8 or 16 bars. Detect these by:
1. Compute RMS energy in 0.5s windows across the full track
2. Find local minima in the energy curve (dips of > 20% below surrounding average)
3. Filter to only boundaries that align with a downbeat timestamp (within 0.2s)
4. These are your crossfade trigger points

**Drop candidate detection:**
A drop candidate is a moment where:
- Energy drops to near-zero for 1–3 seconds (breakdown)
- Followed by a sharp energy spike (the drop itself)
Detect: find segments where energy < 0.2 for > 1s, followed by energy > 0.7 within 4s.

---

### Task 3: `dj_algo.py`

Build the three core DJ functions.

```python
def tempo_match(audio_array, source_bpm: float, target_bpm: float) -> np.ndarray:
    """Time-stretch audio to match target BPM using librosa.effects.time_stretch"""
    stretch_factor = source_bpm / target_bpm
    return librosa.effects.time_stretch(audio_array, rate=stretch_factor)

def find_crossfade_point(
    current_track: dict,  # analyzer output
    next_track: dict      # analyzer output
) -> dict:
    """
    Find the optimal crossfade moment.
    Returns:
    {
      "start_at": 183.4,     # seconds into current track to begin fade out
      "fade_duration": 16.0, # seconds (4 bars at current BPM)
      "cue_next_at": 0.47    # timestamp in next track to start from (first downbeat)
    }
    Rules:
    - start_at must be a phrase_boundary in current_track
    - prefer boundaries in the last 30% of the track
    - cue_next_at = first downbeat of next_track
    - fade_duration = 4 bars = (60/bpm) * 16 seconds
    """

def plan_drop(track: dict) -> list:
    """
    Return list of drop events for this track.
    Each event:
    {
      "type": "cut",           # cut | build | drop | singalong
      "at": 92.4,              # timestamp in seconds
      "duration": 2.0,         # how long the effect lasts
      "description": "Drop incoming — crowd ready"
    }
    """
```

---

### Task 4: `brain.py`

Two separate functions — one for local Ollama, one for Claude API.

```python
# --- Ollama: fast, local, real-time decisions ---
def pick_next_track(
    current_track: dict,
    queue: list[dict],
    vibe_mode: str  # "sangeet" | "club" | "lounge" | "baaraat"
) -> dict:
    """
    Call local Ollama (Mistral) with current track metadata + queue.
    Ask it to pick the best next track from queue based on:
    - BPM proximity (within 10 BPM preferred)
    - Key compatibility (same key or relative major/minor)
    - Energy arc (should energy go up, maintain, or cool down?)
    - Vibe mode constraints
    Returns the selected track dict from queue.
    """

# --- Claude API: heavier set planning ---
def plan_set(
    tracks: list[dict],
    vibe_mode: str,
    duration_minutes: int,
    event_context: str  # e.g. "baaraat procession, outdoor, 200 guests"
) -> dict:
    """
    Call Claude claude-sonnet-4-20250514 with full track list + context.
    Ask Claude to return:
    {
      "set_arc": [
        {"track_index": 2, "position": "opener", "notes": "Start mid-energy"},
        {"track_index": 0, "position": "build", "notes": "Increase BPM here"},
        {"track_index": 4, "position": "peak",   "notes": "This is the main drop moment"},
        {"track_index": 1, "position": "cooldown","notes": "Slow it down, crowd is tiring"}
      ],
      "drop_script": "Baaraat wale, yeh wala sun lo! Ab sab nacho!",
      "singalong_tracks": [0, 4],   # indices of tracks with singalong potential
      "total_duration_estimate": 58
    }
    
    Prompt Claude in Hindi-English mix for Indian event context if vibe_mode is
    "sangeet" or "baaraat".
    """
```

**Claude API call format:**
```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}]
)
```

---

### Task 5: `main.py` — FastAPI Endpoints

```
POST /search          — fetch YouTube stream URL + basic metadata
POST /analyze         — full audio analysis of a stream URL
POST /plan-set        — Claude set arc planning
POST /next-track      — Ollama real-time next track pick
GET  /ws/beats        — WebSocket, streams beat timestamps as track plays
```

All endpoints return JSON. WebSocket pushes beat events every `(60/bpm)` seconds.

---

## Phase 2 — Frontend

### DeckUI.jsx — Main layout

Two virtual decks side by side (Deck A playing, Deck B loaded + ready).
Below them: crossfader slider, master volume, BPM display.
Right panel: Queue with upcoming tracks.
Top bar: VibePicker (Sangeet / Club / Lounge / Baaraat) + event context input.

### useAudioEngine.js — Web Audio API wrapper

```javascript
// Core functions to implement:
loadTrack(streamUrl)          // create AudioBufferSourceNode from stream
play(deckId)                  // start playback
crossfadeTo(deckId, duration) // GainNode fade: deck A out, deck B in
setEQ(deckId, bass, mid, tre) // BiquadFilterNode per band
triggerDrop(deckId)           // mute for 2s, then full volume
stripVocals(deckId)           // toggle Spleeter vocal stem on/off
```

### DropAlert.jsx

When a drop_candidate timestamp is approaching (within 8 seconds):
- Show countdown bar: "DROP IN 8... 7... 6..."
- Flash UI red at 3 seconds
- At 0: cut audio → 2s silence → full blast + confetti particle effect

---

## Phase 3 — Voice Drops

Pre-generate 10 drops using ElevenLabs before the event. Store as MP3 locally.
Trigger them via the drop_script from Claude's set plan.

Example drops to generate:
- "Aye aye aye — yeh wala bada banger hai!"
- "Baaraat wale, sab ready ho jao!"
- "Ab koi nahi rukega — floor pe aa jao!"
- "Slow down... slow down... ABHI NAHI!"
- "Sing it with me — everyone!"

---

## Environment Variables Required

```
ANTHROPIC_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Python Dependencies (`requirements.txt`)

```
fastapi
uvicorn
yt-dlp
librosa
essentia
spleeter
numpy
anthropic
python-dotenv
websockets
httpx
```

---

## How to Give This Spec to Antigravity

Paste the following as your first agent task:

---

**Antigravity Day 1 Prompt:**

> Read the full spec above. Your first task is Phase 1, Task 1 and Task 2 only.
>
> 1. Create the `dropsync/backend/` folder structure
> 2. Build `fetcher.py` with `get_stream_url(query)` exactly as specified
> 3. Build `analyzer.py` with `analyze_track(stream_url)` exactly as specified, including BPM harmonic correction, phrase boundary detection, and drop candidate detection
> 4. Create `requirements.txt` with all dependencies
> 5. Write a test script `test_analysis.py` that calls both functions on the query "Diljit Dosanjh Lover" and prints the full JSON output
>
> Do not build the DJ algorithm or frontend yet. Focus on getting clean, correct audio analysis output first. The phrase boundary and drop detection logic is critical — get it right before moving on.

---

## Build Sequence (Do Not Skip Steps)

1. `fetcher.py` → test with 3 songs manually
2. `analyzer.py` → validate BPM accuracy against known tracks
3. `dj_algo.py` → test crossfade point logic in isolation
4. `brain.py` → test Claude set plan output with 5 tracks
5. `main.py` → wire all endpoints
6. `useAudioEngine.js` → test crossfade in browser alone first
7. Full frontend → connect to backend
8. End-to-end test with a real 30-minute set

---

*Built for Indian event circuit. Optimised for weddings, baraats, sangeets, and house parties.*
*Stack: Python + FastAPI + Librosa + yt-dlp + Ollama + Claude API + React + Web Audio API*
