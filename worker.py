import os
import shutil
import subprocess
import json
import librosa
import numpy as np
import pandas as pd
import essentia.standard as es
import yt_dlp
import redis
import warnings

# Suppress the Essentia/Librosa warnings to keep the terminal clean
warnings.filterwarnings("ignore")

# ==========================================
# 1. INITIALIZE CONNECTIONS & PATHS
# ==========================================
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
QUEUE_KEY = "song_queue"
CSV_FILENAME = "music_features_dataset.csv"

# ==========================================
# 2. THE PIPELINE FUNCTIONS
# ==========================================
def download_audio(query, output_dir="audio_data"):
    print(f"\n[1/4] Downloading Audio: '{query}'")
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch1:{query}", download=True)
        title = info['entries'][0]['title']
        expected_filename = os.path.join(output_dir, f"{title}.mp3")
        return title, expected_filename

def separate_stems(audio_path, output_dir="separated"):
    print(f"[2/4] Running Demucs Source Separation...")
    command = ["demucs", audio_path, "-o", output_dir]
    
    try:
        # capture_output=True grabs the logs invisibly. text=True makes them readable.
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # If Demucs crashes, we print exactly what went wrong!
        print(f"\n🛑 --- DEMUCS CRASH LOG ---")
        print(e.stderr)  # This contains the actual error message
        print(f"---------------------------\n")
        raise e  # Pass the error up so the worker skips to the next song
        
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    stems_path = os.path.join(output_dir, "htdemucs", base_name)
    return stems_path

# ==========================================
# 3. FEATURE EXTRACTION (Your Pipeline)
# ==========================================
def extract_features(track_name, original_audio_path, demucs_stems_path=None):
    """
    Extracts Low, Mid, High, and Production features into a flat dictionary
    suitable for a Pandas DataFrame and Latent Space distance calculations.
    """
    print(f"\n--- Extracting Audio Features for: {track_name} ---")
    features = {'track_name': track_name}
    
    # Load as mono for standard spectral analysis, and stereo for spatial analysis
    y_mono, sr = librosa.load(original_audio_path, mono=True)
    y_stereo, _ = librosa.load(original_audio_path, mono=False)
    
    # [LOW-LEVEL: Timbre, Brightness, Texture]
    mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_var'] = np.var(mfccs[i])
        
    centroids = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
    features['spectral_centroid_mean'] = np.mean(centroids)
    
    flux = librosa.onset.onset_strength(y=y_mono, sr=sr)
    features['spectral_flux_mean'] = np.mean(flux)
    
    zcr = librosa.feature.zero_crossing_rate(y_mono)
    features['zcr_mean'] = np.mean(zcr)

    # [MID-LEVEL: Harmony & Rhythm]
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i, note in enumerate(pitch_classes):
        features[f'chroma_{note}_mean'] = np.mean(chroma[i])
        
    extractor = es.MusicExtractor(lowlevelStats=['mean'], rhythmStats=['mean'], tonalStats=['mean'])
    es_features, _ = extractor(original_audio_path)
    
    features['bpm'] = es_features['rhythm.bpm']
    features['key'] = es_features['tonal.key_edma.key']       
    features['scale'] = es_features['tonal.key_edma.scale']   

    # [HIGH-LEVEL: Vibe & Semantics]
    features['danceability'] = es_features['rhythm.danceability']
    # ==========================================
    # 4. HIGH-LEVEL (Deep Learning / TensorFlow)
    # ==========================================
    # Deep learning models require audio to be downsampled to 16kHz to save compute time
    audio_16k = es.MonoLoader(filename=original_audio_path, sampleRate=16000)()
    
    print("Running Neural Networks... (This takes a few seconds)")

    # --- MODEL 1: Emotion (Valence & Arousal) ---
    try:
        # Load the MusiCNN architecture
        embedding_model = es.TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
        embeddings = embedding_model(audio_16k)

        # load valence/arousal model
        muse_model = es.TensorflowPredict2D(graphFilename="models/muse-msd-musicnn-2.pb", output="model/Identity")
        
        # Run the audio through the network (Outputs a 2D matrix of frame-by-frame predictions)
        muse_predictions = muse_model(embeddings)
        
        # Average the frames to get the global mood of the entire song
        mean_muse = np.mean(muse_predictions, axis=0)
        
        # Note: Check your muse-msd-musicnn-1.json file for the exact index mapping!
        # Assuming Valence is index 0 and Arousal is index 1 (Update these based on your JSON):
        features['valence_pred'] = mean_muse[0]
        features['arousal_pred'] = mean_muse[1]
        
    except Exception as e:
        print(f"Could not load MuSe model: {e}")
        features['valence_pred'] = None
        features['arousal_pred'] = None

    # --- MODEL 2: Instrumentation (MTG-Jamendo) ---
    try:
        # Load the EfficientNet architecture
        embedding_model = es.TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        embeddings = embedding_model(audio_16k)

        # Load Instrument Model
        instrument_model = es.TensorflowPredict2D(graphFilename="models/mtg_jamendo_instrument-discogs-effnet-1.pb")
        # Run the audio through the network
        predictions = instrument_model(embeddings)
        mean_instruments = np.mean(predictions, axis=0)
        
        # We don't want to add 400 different instrument columns to our DataFrame.
        # Instead, let's just grab the Top 3 most prominent instruments detected!
        
        # 1. Load the JSON file to see what the numbers actually mean
        with open("models/mtg_jamendo_instrument-discogs-effnet-1.json", "r") as f:
            instrument_metadata = json.load(f)
            instrument_classes = instrument_metadata['classes']
            
        # 2. Find the indices of the top 5 highest probabilities
        top_5_indices = np.argsort(mean_instruments)[-5:][::-1]
        
        # 3. Save the actual instrument names to our features dictionary
        features['top_instrument_1'] = instrument_classes[top_5_indices[0]]
        features['top_instrument_2'] = instrument_classes[top_5_indices[1]]
        features['top_instrument_3'] = instrument_classes[top_5_indices[2]]
        features['top_instrument_4'] = instrument_classes[top_5_indices[3]]
        features['top_instrument_5'] = instrument_classes[top_5_indices[4]]

    except Exception as e:
        print(f"Could not load Instrument model: {e}")
        features['top_instrument_1'] = None
        features['top_instrument_2'] = None
        features['top_instrument_3'] = None
        features['top_instrument_4'] = None
        features['top_instrument_5'] = None

    # [PRODUCTION: Loudness & Spatial]
    if 'lowlevel.loudness_ebu128.integrated' in es_features.descriptorNames():
        features['lufs_loudness'] = es_features['lowlevel.loudness_ebu128.integrated']
    else:
        features['lufs_loudness'] = 0.0
    
    if y_stereo.ndim == 2:
        mid_signal = (y_stereo[0] + y_stereo[1]) / 2.0
        side_signal = (y_stereo[0] - y_stereo[1]) / 2.0
        features['stereo_width_ratio'] = np.sum(side_signal**2) / (np.sum(mid_signal**2) + 1e-6)
    else:
        features['stereo_width_ratio'] = 0.0

    # [INSTRUMENTATION PROFILES via Demucs]
    if demucs_stems_path and os.path.exists(demucs_stems_path):
        stems = ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']
        energies = {}
        for stem in stems:
            stem_path = os.path.join(demucs_stems_path, stem)
            if os.path.exists(stem_path):
                y_stem, _ = librosa.load(stem_path, mono=True)
                energies[stem.split('.')[0]] = np.sum(y_stem**2)
            else:
                energies[stem.split('.')[0]] = 0.0
                
        total_energy = sum(energies.values()) + 1e-6
        features['vocal_presence_ratio'] = energies['vocals'] / total_energy
        features['drum_presence_ratio'] = energies['drums'] / total_energy
        features['bass_presence_ratio'] = energies['bass'] / total_energy
    else:
        features['vocal_presence_ratio'] = None
        features['drum_presence_ratio'] = None
        features['bass_presence_ratio'] = None
    
    return features

# ==========================================
# 3. THE MAIN WORKER LOOP
# ==========================================
def run_worker():
    print("🎧 Musichead Worker Node Started. Listening to Redis Queue...")
    
    while True:
        # blpop Blocks the script until a song appears in the queue!
        # The moment a song is pushed, it pops it and continues.
        _, search_query = r.blpop(QUEUE_KEY)
        
        print(f"\n========================================")
        print(f"🎵 NEW TASK Picked Up: {search_query}")
        
        try:
            # 1. Download
            track_name, audio_path = download_audio(search_query)
            
            # 2. Separate
            stems_path = separate_stems(audio_path)
            
            # 3. Extract
            song_features = extract_features(track_name, audio_path, stems_path)
            
            # 4. Save to CSV
            print(f"[4/4] Saving to CSV and Cleaning up...")
            df = pd.DataFrame([song_features])
            file_exists = os.path.isfile(CSV_FILENAME)
            df.to_csv(CSV_FILENAME, mode='a', header=not file_exists, index=False)
            
            # 5. NUCLEAR CLEANUP (Keep storage cost at $0)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(stems_path):
                shutil.rmtree(stems_path) # Deletes the whole song folder inside htdemucs
                
            print(f"✅ SUCCESSFULLY PROCESSED & CLEANED: {track_name}")
            
        except Exception as e:
            print(f"❌ ERROR processing '{search_query}': {e}")
            # Optional: Push it to a "failed_jobs" Redis queue here if you want to retry later

if __name__ == "__main__":
    run_worker()