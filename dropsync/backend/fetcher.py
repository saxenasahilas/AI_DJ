import yt_dlp

def get_stream_url(query: str) -> dict:
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Search and extract info
        info = ydl.extract_info(f"ytsearch1:{query}", download=False)
        
        if not info or ('entries' in info and len(info['entries']) == 0):
            raise ValueError(f"No results for: {query}")

        if 'entries' in info and len(info['entries']) > 0:
            entry = info['entries'][0]
        else:
            entry = info
            
        stream_url = entry.get('url')
        
        formats = entry.get('formats', [])
        audio_formats = [
            f for f in formats
            if f.get('acodec') != 'none' and f.get('vcodec') == 'none'
        ]
        if audio_formats:
            # Pick highest quality audio-only
            best = max(audio_formats, key=lambda f: f.get('abr') or 0)
            stream_url = best.get('url')
                    
        return {
            "url": stream_url,
            "webpage_url": entry.get('webpage_url', ''),
            "title": entry.get('title', 'Unknown Title'),
            "duration": entry.get('duration', 0),
            "thumbnail": entry.get('thumbnail', '')
        }
