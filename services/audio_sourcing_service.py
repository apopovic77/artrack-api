import os
try:
    import freesound
except ModuleNotFoundError:
    freesound = None
import httpx

async def find_sfx_on_freesound(description: str) -> str | None:
    """
    Searches for a sound effect on Freesound.org and returns a download URL.
    """
    print(f"--- Audio Sourcing: Searching Freesound for SFX: '{description}'")
    try:
        if freesound is None:
            raise RuntimeError("Freesound support requires the 'freesound' package. Install it with 'pip install freesound'.")
        client = freesound.FreesoundClient()
        client.set_token(os.getenv("FREESOUND_CLIENT_SECRET"), "token")
        
        results = client.text_search(query=description, filter="license:\"Creative Commons 0\" duration:[0.5 TO 15]", fields="id,name,previews")
        
        if results.count > 0:
            sound = results[0]
            return sound.previews.preview_hq_mp3
        else:
            return None
    except Exception as e:
        print(f"--- ERROR [Freesound SFX]: {e}")
        return None

async def find_music_on_freesound(description: str) -> str | None:
    """
    Searches for music on Freesound.org and returns a download URL.
    """
    print(f"--- Audio Sourcing: Searching Freesound for Music: '{description}'")
    try:
        if freesound is None:
            raise RuntimeError("Freesound support requires the 'freesound' package. Install it with 'pip install freesound'.")
        client = freesound.FreesoundClient()
        client.set_token(os.getenv("FREESOUND_CLIENT_SECRET"), "token")
        
        results = client.text_search(query=description, filter="license:\"Creative Commons 0\" duration:[30 TO 300]", fields="id,name,previews")
        
        if results.count > 0:
            sound = results[0]
            return sound.previews.preview_hq_mp3
        else:
            return None
    except Exception as e:
        print(f"--- ERROR [Freesound Music]: {e}")
        return None
