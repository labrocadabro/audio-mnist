import io
import librosa
import numpy as np
from PIL import Image
from librosa_config import config, MAX_WIDTH


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
    buffer = io.BytesIO()
    img.save(buffer, format="png")
    buffer.seek(0)
    return buffer.getvalue()


def convert(audio_file):
    img_bytes = spectrogram_image(audio_file)
    buffer = io.BytesIO(img_bytes)
    img = Image.open(buffer)
    return img
