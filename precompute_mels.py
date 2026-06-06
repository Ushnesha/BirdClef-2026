import os
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import librosa
from tqdm import tqdm
import pickle

# ─────────────────────────────────────────
# CONFIG (must match train.py)
# ─────────────────────────────────────────
class CFG:
    data_dir     = "/kaggle/input/birdclef-2026"
    sample_rate  = 32000
    duration     = 5
    n_mels       = 128
    fmin         = 50
    fmax         = 14000
    n_fft        = 1024
    hop_length   = 320
    mel_cache_dir = "mel_cache"  # where to save precomputed mels


# ─────────────────────────────────────────
# AUDIO UTILS (same as train.py)
# ─────────────────────────────────────────
def audio_to_melspec_from_array(y, sr=CFG.sample_rate):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax,
        n_fft=CFG.n_fft, hop_length=CFG.hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db.astype(np.float32)


def load_audio_chunk(path, sr=CFG.sample_rate, duration=CFG.duration):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


# ─────────────────────────────────────────
# PRECOMPUTE
# ─────────────────────────────────────────
def precompute_mels():
    # Create cache directory
    os.makedirs(CFG.mel_cache_dir, exist_ok=True)

    # Load metadata
    df = pd.read_csv(os.path.join(CFG.data_dir, "train.csv"))
    print(f"Total samples to precompute: {len(df)}")

    # Precompute each audio file
    failed = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing mels"):
        filename = row["filename"]
        cache_path = os.path.join(CFG.mel_cache_dir, filename.replace(".ogg", ".npy"))

        # Create subdirectory if needed
        cache_subdir = os.path.dirname(cache_path)
        os.makedirs(cache_subdir, exist_ok=True)

        try:
            # Skip if already cached
            if os.path.exists(cache_path):
                continue

            # Load and process audio
            audio_path = os.path.join(CFG.data_dir, "train_audio", filename)
            y = load_audio_chunk(audio_path)
            mel = audio_to_melspec_from_array(y)

            # Save as .npy
            np.save(cache_path, mel)

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            failed.append(filename)

    print(f"\nPrecompute complete!")
    print(f"Cached to: {CFG.mel_cache_dir}/")
    if failed:
        print(f"Failed to process {len(failed)} files: {failed[:5]}")


if __name__ == "__main__":
    precompute_mels()
