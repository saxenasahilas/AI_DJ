import librosa
import numpy as np
import tempfile
import urllib.request
import shutil
import os

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

def analyze_track(stream_url: str) -> dict:
    if MOCK_MODE:
        return {
            "bpm": 128.0,
            "bpm_confidence": 0.85,
            "key": "F#",
            "beats": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "downbeats": [0.5, 2.5, 4.5],
            "phrase_boundaries": [16.5, 32.5, 64.5],
            "energy_curve": [0.2, 0.5, 0.8, 0.9, 0.3],
            "mood": "energetic",
            "danceability": 0.88,
            "drop_candidates": [32.5, 64.5]
        }

    """
    Analyzes an audio stream from a given URL and returns its features.
    Features include BPM, beat grids, energy curves, phrase boundaries, and drop candidates.
    """
    # 1. Download to a temporary file
    # We use a user agent because some raw URLs might block default urllib agents
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.m4a')
    req = urllib.request.Request(
        stream_url,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            with open(temp_file.name, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        
        # Load audio using librosa (using 22050 Hz for performance and standard analysis)
        y, sr = librosa.load(temp_file.name, sr=22050)
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    # 2. Extract BPM and Beats
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units='frames')
    
    # Confidence can be estimated from onset strength at beat positions
    bpm_confidence = float(np.mean(onset_env[beat_frames]) / np.max(onset_env)) if len(beat_frames) > 0 else 0.5
    
    # Harmonic correction for BPM
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    if bpm < 80:
        bpm *= 2
    elif bpm > 180:
        bpm /= 2
        
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Calculate downbeats (Assuming 4/4 time signature)
    # Simple heuristic: every 4th beat is a downbeat starting from the first beat.
    downbeats = beat_times[::4].tolist()
    
    # 3. RMS Energy and Phrase Boundaries
    # Compute RMS energy in 0.5s windows
    hop_length = int(sr * 0.5)
    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Phrase boundaries: Local minima in energy curve (dips of > 20% below surrounding average)
    phrase_boundaries = []
    window_size = 10  # 5 seconds surrounding average (5s / 0.5s = 10 frames)
    
    for i in range(len(rms)):
        start = max(0, i - window_size)
        end = min(len(rms), i + window_size + 1)
        surrounding_avg = np.mean(rms[start:end])
        
        # Check for > 20% dip
        if rms[i] < surrounding_avg * 0.8:
            # Check if it's a local minimum in a small neighborhood (e.g., 3 frames)
            local_start = max(0, i - 1)
            local_end = min(len(rms), i + 2)
            if rms[i] == np.min(rms[local_start:local_end]):
                boundary_time = rms_times[i]
                
                # Filter to only boundaries that align with a downbeat timestamp (within 0.5s)
                if len(downbeats) > 0:
                    nearest_downbeat = min(downbeats, key=lambda d: abs(d - boundary_time))
                    if abs(nearest_downbeat - boundary_time) <= 0.5:
                        phrase_boundaries.append(nearest_downbeat)

    phrase_boundaries = sorted(list(set(phrase_boundaries)))  # Unique phrase boundaries
    
    # Normalize energy curve to 0-1
    max_rms = np.max(rms) if np.max(rms) > 0 else 1
    energy_curve = (rms / max_rms).tolist()
    
    # 4. Drop Candidate Detection
    # Detect: find segments where energy < 0.2 for > 1s, followed by energy > 0.7 within 4s.
    drop_candidates = []
    i = 0
    while i < len(energy_curve) - 8:
        if energy_curve[i] < 0.2:
            # Check if it stays low for > 1s (2 frames of 0.5s each)
            if energy_curve[i+1] < 0.2 and energy_curve[i+2] < 0.2:
                # Search for spike > 0.7 within next 4s (8 frames)
                for j in range(i + 3, min(i + 11, len(energy_curve))):
                    if energy_curve[j] > 0.7:
                        drop_candidates.append(float(rms_times[j]))
                        i = j  # Skip forward
                        break
        i += 1
                        
    # 5. Key Detection (Simple heuristic using chroma)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chromagram, axis=1)
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = np.argmax(mean_chroma)
    key = chroma_to_key[key_idx]
    
    # 6. Mood and Danceability (Heuristics based on BPM and Energy)
    danceability = min(1.0, max(0.0, (bpm - 60) / 100 * 0.5 + np.mean(energy_curve) * 0.5))
    if np.mean(energy_curve) > 0.6 and bpm > 110:
        mood = "energetic"
    elif np.mean(energy_curve) < 0.4 and bpm < 100:
        mood = "calm"
    elif np.max(energy_curve) > 0.8:
        mood = "aggressive"
    else:
        mood = "melancholic"

    return {
        "bpm": float(round(bpm, 2)),
        "bpm_confidence": float(round(bpm_confidence, 2)),
        "key": key,
        "beats": [float(round(b, 2)) for b in beat_times],
        "downbeats": [float(round(d, 2)) for d in downbeats],
        "phrase_boundaries": [float(round(p, 2)) for p in phrase_boundaries],
        "energy_curve": [float(round(e, 3)) for e in energy_curve],
        "mood": mood,
        "danceability": float(round(danceability, 2)),
        "drop_candidates": [float(round(d, 2)) for d in drop_candidates]
    }
