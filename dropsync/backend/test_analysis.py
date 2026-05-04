import json
from fetcher import get_stream_url
from analyzer import analyze_track

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
        analysis_result = analyze_track(track_info.get("webpage_url"))
        print("\n--- Final Analysis JSON Output ---")
        print(json.dumps(analysis_result, indent=2))
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
