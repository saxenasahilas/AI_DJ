import json
from fetcher import get_stream_url
from analyzer import analyze_track
from dj_algo import find_crossfade_point, plan_drops, score_next_track
from brain import pick_next_track, plan_set_local

def main():
    query = "Diljit Dosanjh Lover"
    print(f"Fetching stream URL for: '{query}'...")
    track_info = get_stream_url(query)
    
    stream_url = track_info.get("url")
    if not stream_url:
        print("Failed to get stream URL.")
        print("Output:", track_info)
        return
        
    print(f"Found stream URL for: {track_info.get('title')}")
    print(f"Duration: {track_info.get('duration')} seconds")
    print("Analyzing track (this may take a minute depending on network and CPU)...")
    
    try:
        analysis_result = analyze_track(track_info.get("url"))
        print("\n--- Final Analysis JSON Output ---")
        print(json.dumps(analysis_result, indent=2))

        # Simulate two tracks using mock data
        track_a = analysis_result
        track_b = {**analysis_result, "bpm": 132.0, "key": "G"}
        track_a["duration"] = 187

        print("\n--- Crossfade Point ---")
        print(json.dumps(find_crossfade_point(track_a, track_b), indent=2))

        print("\n--- Drop Plan ---")
        print(json.dumps(plan_drops(track_a), indent=2))

        print("\n--- Next Track Score ---")
        print(score_next_track(track_a, track_b, vibe_mode="club", position_in_set=0.4))

        queue = [
            {**track_b, "title": "Pasoori"},
            {**track_a, "bpm": 118.0, "title": "Tum Se Hi", "mood": "calm"},
            {**track_b, "bpm": 135.0, "title": "Excuses", "mood": "energetic"}
        ]

        print("\n--- Next Track Pick ---")
        pick = pick_next_track(track_a, queue, vibe_mode="sangeet", position_in_set=0.3)
        print(json.dumps(pick, indent=2))

        print("\n--- Set Plan (Local) ---")
        plan = plan_set_local(
            [track_a, track_b] + queue,
            vibe_mode="baaraat",
            duration_minutes=45,
            event_context="Baaraat procession, outdoor, 300 guests, Bareilly UP"
        )
        print(json.dumps(plan, indent=2))
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
