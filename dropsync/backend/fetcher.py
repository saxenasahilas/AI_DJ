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
        
        if 'entries' in info and len(info['entries']) > 0:
            entry = info['entries'][0]
        else:
            entry = info
            
        stream_url = entry.get('url')
        
        # If stream_url seems to be the main youtube video URL,
        # we try to find the actual stream url from the 'formats'
        if stream_url and "youtube.com/watch" in stream_url:
            for f in entry.get('formats', []):
                # Look for an audio-only format
                if f.get('acodec') != 'none' and f.get('vcodec') == 'none':
                    stream_url = f.get('url')
                    break
                    
        return {
            "url": stream_url,
            "title": entry.get('title', 'Unknown Title'),
            "duration": entry.get('duration', 0),
            "thumbnail": entry.get('thumbnail', '')
        }
