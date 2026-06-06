import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 1. Define matching configuration (from train.py)
class CFG:
    sample_rate  = 32000
    n_mels       = 128
    fmin         = 50
    fmax         = 14000
    n_fft        = 1024
    hop_length   = 320

# 2. Load the audio file
audio_path = "train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
y, sr = librosa.load(audio_path, sr=CFG.sample_rate)

# 3. Compute Mel-Spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=y, 
    sr=sr, 
    n_mels=CFG.n_mels, 
    fmin=CFG.fmin, 
    fmax=CFG.fmax,
    n_fft=CFG.n_fft, 
    hop_length=CFG.hop_length
)

# 4. Convert power spectrogram to decibels (log scale)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# 5. Plot Mel-Spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(
    mel_spec_db,
    sr=sr,
    hop_length=CFG.hop_length,
    x_axis='time',
    y_axis='mel',
    fmin=CFG.fmin,
    fmax=CFG.fmax,
    cmap='magma' # 'magma' or 'viridis' are recommended for audio representation
)

plt.colorbar(format='%+2.0f dB')
plt.title(f"Mel Spectrogram - {os.path.basename(audio_path)}")
plt.tight_layout()
plt.show()
