import requests

def is_video_url_reachable(url: str) -> bool:
    """
    Check if a video URL is reachable by sending a HEAD request.
    Returns True if status code is 200, False otherwise.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except Exception:
        return False
