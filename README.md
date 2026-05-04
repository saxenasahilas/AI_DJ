# DropSync — AI DJ MVP

An AI-powered DJ engine that fetches audio from YouTube, analyzes beats in real time, and mixes tracks like a live DJ using Ollama / Claude for set intelligence. Built for the Indian event circuit.

## Features (Phase 1 Backend)
- **Audio Fetching**: Directly streams audio from YouTube via `yt-dlp`.
- **Beat Analysis**: Extracts BPM, harmonic corrections, phrase boundaries, and identifies drop candidates using `librosa`.
- **Local AI Brain**: Uses local `llama3.1` and `mistral` (via Ollama) to dynamically plan set arcs, drop scripts, and pick the next track based on vibe mode and energy curves.
- **FastAPI Core**: Ready-to-go FastAPI backend meant to hook into the React frontend Deck UI.

## Setup & Running

1. **Install dependencies:**
   ```bash
   cd dropsync/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Ollama (for the local AI Brain):**
   Make sure you have Ollama installed locally, then pull the necessary models:
   ```bash
   ollama pull llama3.1
   ollama pull mistral
   ```

3. **Configure Environment:**
   If you wish to use Claude instead of local Ollama, update `.env` in `dropsync/backend/` with your `ANTHROPIC_API_KEY`.

4. **Run Backend:**
   ```bash
   cd dropsync/backend
   uvicorn main:app --reload
   ```

## Architecture
See `ai_dj_mvp_spec.md` for full implementation details, frontend Web Audio API structure, and future Phase instructions.
