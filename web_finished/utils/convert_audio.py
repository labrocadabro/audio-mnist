import io
import librosa
import numpy as np
from PIL import Image


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def pad_array(arr, target_shape):
    pad_width = [(0, max(0, target_shape[i] - arr.shape[i])) for i in range(arr.ndim)]
    return np.pad(arr, pad_width, mode="constant")


def resize_array(arr, new_shape):
    _, width = arr.shape
    _, max_width = new_shape
    return arr[:, (width - max_width) :]


def normalize_width(arr):
    height, width = arr.shape
    max_width = 25
    if width > max_width:
        arr = resize_array(arr, (height, max_width))
    elif width < max_width:
        arr = pad_array(arr, (height, max_width))
    return arr


def spectrogram_image(audio_file):
    hop_length = 512  # number of samples per time-step in spectrogram
    n_mels = 128  # number of bins in spectrogram. Height of image
    time_steps = 512  # number of time-steps. Width of image

    y, sr = librosa.load(audio_file, duration=1.0)

    # extract a fixed length window
    start_sample = 0  # starting at beginning
    length_samples = time_steps * hop_length
    window = y[start_sample : start_sample + length_samples]

    mels = librosa.feature.melspectrogram(
        y=window, sr=sr, n_mels=n_mels, n_fft=hop_length * 2, hop_length=hop_length
    )
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img_arr = scale_minmax(mels, 0, 255).astype(np.uint8)
    img_arr = normalize_width(img_arr)
    print(img_arr.shape)
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
