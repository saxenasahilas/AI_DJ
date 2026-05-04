import numpy as np
import librosa

# ─────────────────────────────────────────
# 1. TEMPO MATCH
# ─────────────────────────────────────────
def tempo_match(y: np.ndarray, source_bpm: float, target_bpm: float) -> np.ndarray:
    """
    Time-stretch audio array to match target BPM.
    Returns stretched audio array.
    """
    if source_bpm <= 0 or target_bpm <= 0:
        return y
    stretch_rate = target_bpm / source_bpm
    # Clamp stretch to avoid extreme distortion (max 20% speed change)
    stretch_rate = np.clip(stretch_rate, 0.8, 1.2)
    return librosa.effects.time_stretch(y, rate=float(stretch_rate))


# ─────────────────────────────────────────
# 2. FIND CROSSFADE POINT
# ─────────────────────────────────────────
def find_crossfade_point(current_track: dict, next_track: dict) -> dict:
    """
    Find optimal moment to begin crossfade from current to next track.

    Rules:
    - start_at must be a phrase_boundary in the last 40% of current track
    - fade_duration = 4 bars = (60/bpm) * 16 seconds
    - cue_next_at = first downbeat of next track
    - fallback: if no valid phrase boundary found, use 85% of duration
    """
    bpm = current_track.get('bpm', 128.0)
    duration = current_track.get('duration', 180)
    phrase_boundaries = current_track.get('phrase_boundaries', [])
    next_downbeats = next_track.get('downbeats', [0.5])

    fade_duration = (60.0 / bpm) * 16  # 4 bars

    # Find phrase boundaries in last 40% of track
    cutoff = duration * 0.60
    late_boundaries = [b for b in phrase_boundaries if b >= cutoff]

    if late_boundaries:
        # Pick the one that leaves enough room for the full fade
        valid = [b for b in late_boundaries if b + fade_duration <= duration]
        start_at = valid[0] if valid else late_boundaries[0]
    else:
        # Fallback: 85% of track duration, snapped to nearest beat
        start_at = duration * 0.85

    cue_next_at = next_downbeats[0] if next_downbeats else 0.5

    return {
        "start_at": float(round(start_at, 2)),
        "fade_duration": float(round(fade_duration, 2)),
        "cue_next_at": float(round(cue_next_at, 2))
    }


# ─────────────────────────────────────────
# 3. PLAN DROPS
# ─────────────────────────────────────────
def plan_drops(track: dict) -> list:
    """
    Return list of drop events for a track.

    Event types:
    - 'cut'       : silence for 1-2s then full volume smash
    - 'build'     : gradual energy rise over 8 bars leading to drop
    - 'drop'      : the actual peak moment
    - 'singalong' : strip beat slightly, crowd fills in vocals
    """
    events = []
    bpm = track.get('bpm', 128.0)
    drop_candidates = track.get('drop_candidates', [])
    phrase_boundaries = track.get('phrase_boundaries', [])
    duration = track.get('duration', 180)
    bar_duration = (60.0 / bpm) * 4  # one bar in seconds

    for drop_time in drop_candidates:
        # Build event starts 8 bars before drop
        build_time = max(0, drop_time - (bar_duration * 8))

        events.append({
            "type": "build",
            "at": float(round(build_time, 2)),
            "duration": float(round(bar_duration * 8, 2)),
            "description": "Energy rising — crowd ready"
        })
        events.append({
            "type": "cut",
            "at": float(round(drop_time - 2.0, 2)),
            "duration": 2.0,
            "description": "Beat cut — silence before drop"
        })
        events.append({
            "type": "drop",
            "at": float(round(drop_time, 2)),
            "duration": float(round(bar_duration * 4, 2)),
            "description": "DROP — full blast"
        })

    # Singalong: tag the phrase boundary nearest to 60% of track
    # (typically where a chorus repeats and crowd knows the words)
    if phrase_boundaries:
        target = duration * 0.60
        nearest = min(phrase_boundaries, key=lambda b: abs(b - target))
        events.append({
            "type": "singalong",
            "at": float(round(nearest, 2)),
            "duration": float(round(bar_duration * 8, 2)),
            "description": "Singalong window — drop beat slightly"
        })

    # Sort all events by timestamp
    events.sort(key=lambda e: e['at'])
    return events


# ─────────────────────────────────────────
# 4. KEY COMPATIBILITY CHECK
# ─────────────────────────────────────────
def keys_are_compatible(key_a: str, key_b: str) -> bool:
    """
    Check if two musical keys are compatible for mixing.
    Uses Camelot wheel logic — adjacent keys mix well.
    Compatible = same key, relative major/minor, or one step on Camelot wheel.
    """
    # Camelot wheel mapping (key → camelot number)
    camelot = {
        'C': 8, 'G': 9, 'D': 10, 'A': 11, 'E': 12,
        'B': 1, 'F#': 2, 'C#': 3, 'G#': 4, 'D#': 5,
        'A#': 6, 'F': 7,
        # Minor equivalents (same numbers, different suffix)
        'Am': 8, 'Em': 9, 'Bm': 10, 'F#m': 11, 'C#m': 12,
        'G#m': 1, 'D#m': 2, 'A#m': 3, 'Fm': 4, 'Cm': 5,
        'Gm': 6, 'Dm': 7
    }

    # Normalize key strings
    key_a = key_a.replace(' minor', 'm').replace(' major', '').strip()
    key_b = key_b.replace(' minor', 'm').replace(' major', '').strip()

    num_a = camelot.get(key_a)
    num_b = camelot.get(key_b)

    if num_a is None or num_b is None:
        return True  # Unknown key — allow mix, don't block

    # Compatible if same number or adjacent on wheel (circular 1-12)
    diff = abs(num_a - num_b)
    return diff <= 1 or diff == 11  # 11 = wrapping around (12→1)


# ─────────────────────────────────────────
# 5. SCORE NEXT TRACK
# ─────────────────────────────────────────
def score_next_track(
    current: dict,
    candidate: dict,
    vibe_mode: str,
    position_in_set: float  # 0.0 = start, 1.0 = end
) -> float:
    """
    Score a candidate track for how well it follows the current track.
    Higher score = better next track.
    Used by pick_next_track in brain.py as a fallback / validator.
    """
    score = 0.0

    # BPM proximity (max 40 points)
    bpm_diff = abs(current.get('bpm', 128) - candidate.get('bpm', 128))
    bpm_score = max(0, 40 - bpm_diff * 2)
    score += bpm_score

    # Key compatibility (30 points)
    if keys_are_compatible(
        current.get('key', 'C'),
        candidate.get('key', 'C')
    ):
        score += 30

    # Energy arc (30 points)
    # Early in set: prefer similar or higher energy
    # Late in set: prefer lower energy (cooldown)
    current_energy = np.mean(current.get('energy_curve', [0.5]))
    candidate_energy = np.mean(candidate.get('energy_curve', [0.5]))
    energy_diff = candidate_energy - current_energy

    if position_in_set < 0.7:
        # Build phase: reward upward energy
        energy_score = 30 if energy_diff >= 0 else max(0, 30 + energy_diff * 30)
    else:
        # Cooldown phase: reward downward energy
        energy_score = 30 if energy_diff <= 0 else max(0, 30 - energy_diff * 30)

    score += energy_score

    return round(score, 2)
