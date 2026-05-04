import json
from fetcher import get_stream_url
from analyzer import analyze_track
from dj_algo import find_crossfade_point, plan_drops, score_next_track

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
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
