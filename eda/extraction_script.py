import os
import subprocess
import librosa
import numpy as np
import pandas as pd
import essentia.standard as es
import yt_dlp
import json

# ==========================================
# 1. AUTOMATED DOWNLOAD (yt-dlp)
# ==========================================
def download_audio(query, output_dir="audio_data"):
    """Searches YouTube for the query, downloads the best audio, and converts to MP3."""
    print(f"\n--- Searching and Downloading: '{query}' ---")
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # Saves file as exactly "Song Title.mp3"
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'quiet': False
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # ytsearch1: strictly grabs the top 1 search result
        info = ydl.extract_info(f"ytsearch1:{query}", download=True)
        title = info['entries'][0]['title']
        
        # Determine the final file path (yt-dlp converts the extension to .mp3)
        expected_filename = os.path.join(output_dir, f"{title}.mp3")
        
        print(f"Downloaded successfully: {expected_filename}")
        return title, expected_filename

# ==========================================
# 2. AUTOMATED STEM SEPARATION (Demucs)
# ==========================================
def separate_stems(audio_path, output_dir="separated"):
    """Triggers the Demucs CLI command via Python to isolate the audio stems."""
    print(f"\n--- Running Demucs Source Separation ---")
    
    # Notice we removed the '--mp3' flag! 
    # This forces Demucs to output .wav files, which matches your extraction script perfectly.
    command = ["demucs", audio_path]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Demucs encountered an error: {e}")
        return None
        
    # Demucs creates the folder: separated/htdemucs/<filename_without_extension>
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    stems_path = os.path.join(output_dir, "htdemucs", base_name)
    
    print(f"Stems successfully isolated at: {stems_path}")
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
# 4. ORCHESTRATOR & APPEND TO CSV
# ==========================================
if __name__ == "__main__":
    
    # 1. DEFINE YOUR SONG SEARCH QUERY HERE
    search_query = "A$AP Rocky Stole Ya Flow"
    csv_filename = "music_features_dataset.csv"
    
    try:
        # Step 1: Download
        track_name, audio_path = download_audio(search_query)
        
        # Step 2: Separate Stems
        stems_path = separate_stems(audio_path)
        
        # Step 3: Extract Features
        song_data = extract_features(
            track_name=track_name,
            original_audio_path=audio_path,
            demucs_stems_path=stems_path
        )
        
        # Step 4: Append to DataFrame and save
        df = pd.DataFrame([song_data])
        
        # Check if file exists to determine if we need to write the header row
        file_exists = os.path.isfile(csv_filename)
        
        # mode='a' appends to the file instead of overwriting!
        df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
        
        print(f"\n✅ SUCCESS! Features for '{track_name}' appended to {csv_filename}")
        print(df[['track_name', 'bpm', 'key', 'danceability', 'lufs_loudness']])
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")