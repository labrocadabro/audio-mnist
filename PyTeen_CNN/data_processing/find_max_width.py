import librosa
import numpy as np
from librosa_config import config
from pathlib import Path
from collections import defaultdict


def spectrogram_image(audio_file):
    y, sr = librosa.load(audio_file)
    mels = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=config["n_mels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        fmin=config["fmin"],
        fmax=config["fmax"],
    )
    img_arr = scale_minmax(mels, 0, 255).astype(np.uint8)
    img_arr = np.flip(img_arr, axis=0)  # put low frequencies at the bottom in image
    img_arr = 255 - img_arr  # invert. make black==more energy
    _, img_width = img_arr.shape
    return img_width


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def weighted_average(dictionary):
    total_sum = 0
    weighted_sum = 0

    for key, frequency in dictionary.items():
        total_sum += key * frequency
        weighted_sum += frequency

    if weighted_sum == 0:
        return None  # Handle division by zero error if no frequencies are provided
    else:
        return total_sum / weighted_sum


p = Path(__file__).parent
in_path = p.parent / "data/audio/"

pathlist = in_path.rglob("**/*.wav")

widths = defaultdict(int)

for path in pathlist:
    with open(path, "rb") as f:
        width = spectrogram_image(f)
        widths[width] += 1

print("Width")
print(widths)
print("Max width:", max(widths.keys()))
print("Mode:", max(widths, key=widths.get))
print("Weighted average:", weighted_average(widths))
print()
