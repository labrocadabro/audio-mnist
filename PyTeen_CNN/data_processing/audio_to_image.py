import librosa
from librosa_config import config, MAX_WIDTH
import os
from pathlib import Path
import librosa
import numpy as np
from PIL import Image

p = Path(__file__).parent.parent
in_path = p / "data/audio/"
out_path = (
    p / f'data/images/{config["hop_length"]}-{config["n_fft"]}-{config["n_mels"]}'
)

pathlist = in_path.rglob("**/*.wav")


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def normalize_width(arr):
    _, current_width = arr.shape
    pad_width = max(0, MAX_WIDTH - current_width)
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    padded_arr = np.pad(arr, ((0, 0), (left_pad, right_pad)), mode="constant")
    return padded_arr


def spectrogram_image(audio_file):
    y, sr = librosa.load(audio_file)
    mels = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=config["n_mels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )
    mels_decibels = librosa.power_to_db(mels, ref=np.max)

    img_arr = scale_minmax(mels_decibels, 0, 255).astype(np.uint8)
    img_arr = normalize_width(img_arr)
    img_arr = np.flip(img_arr, axis=0)  # put low frequencies at the bottom in image
    img_arr = 255 - img_arr  # invert. make black==more energy

    # save as PNG
    img = Image.fromarray(img_arr)
    return img


for i, path in enumerate(pathlist):
    digit_folder = f"{out_path}/{path.parent.parent.name}/{path.parent.name}"
    if not os.path.exists(digit_folder):
        os.makedirs(digit_folder)
    with open(path, "rb") as f:
        img = spectrogram_image(f)

    png_filename = f"{digit_folder}/{path.stem}.png"
    img.save(png_filename)
