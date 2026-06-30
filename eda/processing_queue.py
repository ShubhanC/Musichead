import redis
from dotenv import load_dotenv
import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ==========================================
# 1. INITIALIZE REDIS CONNECTION
# ==========================================
# decode_responses=True ensures we get normal Python strings back instead of byte strings
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Define our Redis keys
QUEUE_KEY = "song_queue"      # A Redis LIST (Where workers pull from)
INDEX_KEY = "seen_songs"      # A Redis SET  (O(1) lookup to prevent duplicates)
#r.delete(INDEX_KEY)  # Clear the queue for fresh testing

# ==========================================
# 2. THE PRODUCER LOGIC
# ==========================================
def ingest_spotify_playlist(playlist_data):
    """
    Parses a Spotify API playlist payload, formats the search query, 
    checks for duplicates, and pushes to the Redis queue.
    """
    
    print(f"\n--- Ingesting Playlist: '{playlist_data['name']}' ---")
    
    # Navigate down to the actual track items based on your payload
    tracks = playlist_data['items']['items']
    queued_count = 0
    skipped_count = 0
    
    for item in tracks:
        track = item['item']
        
        # Extract the track name and primary artist
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        
        # THE HACK: Append 'official audio' so yt-dlp avoids the music videos!
        search_query = f"{artist_name} - {track_name} official audio"
        
        # ---------------------------------------------------------
        # THE GATEKEEPER: O(1) Duplicate Check
        # ---------------------------------------------------------
        # sismember checks if the query exists in our 'seen_songs' SET
        if r.sismember(INDEX_KEY, search_query):
            print(f"⏭️  Skipped (Already indexed): {search_query}")
            skipped_count += 1
            continue
            
        # ---------------------------------------------------------
        # PUSH TO BROKER
        # ---------------------------------------------------------
        # 1. rpush (Right Push) adds it to the back of our task queue
        r.rpush(QUEUE_KEY, search_query)
        
        # 2. sadd (Set Add) marks it as 'seen' so we never queue it again
        r.sadd(INDEX_KEY, search_query)
        
        print(f"✅ Queued: {search_query}")
        queued_count += 1
        
    print(f"\nFinished: Queued {queued_count} new songs. Skipped {skipped_count} duplicates.")

# ==========================================
# RUNNING THE PRODUCER
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
    SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
    SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    scope = ["user-library-read", "playlist-read-collaborative", "playlist-read-private"]

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope=scope))
    
    # Assuming 'spotify_payload' is the dictionary you pasted above
    # https://open.spotify.com/playlist/4E0uHZYTeyablDcenbJ4De?si=fa3cc3f3789c4c29
    spotify_payload = sp.playlist("spotify:playlist:4E0uHZYTeyablDcenbJ4De")
    
    ingest_spotify_playlist(spotify_payload)
    
    # --- HELPER COMMANDS FOR TESTING ---
    # To check how many items are currently waiting in your queue:
    queue_length = r.llen(QUEUE_KEY)
    print(f"\nCurrent Queue Backlog: {queue_length} songs waiting to be processed.")