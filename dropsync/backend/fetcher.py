import httpx
import os

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

def get_stream_url(query: str) -> dict:
    if MOCK_MODE:
        return {
            "url": "https://mock-stream-url.com/audio.mp3",
            "webpage_url": "https://youtube.com/watch?v=mock_id",
            "title": f"Mocked Result for: {query}",
            "duration": 210,
            "thumbnail": "https://mock-thumbnail.com/img.jpg"
        }

    instances = [
        "https://inv.nadeko.net",
        "https://invidious.nerdvpn.de",
        "https://yt.drgnz.club"
    ]
    
    for BASE in instances:
        try:
            # Search
            search_response = httpx.get(f"{BASE}/api/v1/search", params={
                "q": query, "type": "video", "page": 1
            }, timeout=10)
            
            if search_response.status_code != 200:
                continue
                
            search = search_response.json()
            if not search:
                continue
                
            video_id = search[0]['videoId']
            title = search[0]['title']
            duration = search[0]['lengthSeconds']
            
            # Get streams
            video_response = httpx.get(f"{BASE}/api/v1/videos/{video_id}", timeout=10)
            if video_response.status_code != 200:
                continue
                
            video = video_response.json()
            
            # Pick best audio-only format
            audio_formats = [
                f for f in video.get('adaptiveFormats', [])
                if 'audio' in f.get('type', '') and 'video' not in f.get('type', '')
            ]
            
            if not audio_formats:
                continue
                
            best = max(audio_formats, key=lambda f: int(f.get('bitrate', 0)))
            
            return {
                "url": best['url'],
                "webpage_url": f"https://youtube.com/watch?v={video_id}",
                "title": title,
                "duration": duration,
                "thumbnail": f"{BASE}/vi/{video_id}/maxres.jpg"
            }
        except Exception:
            continue
            
    raise ValueError(f"Could not fetch video '{query}' from any Invidious instance")
