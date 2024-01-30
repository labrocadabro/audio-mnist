import librosa
from pathlib import Path


def energy_duration(audio_file):
    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)
    energy = sum(abs(y) ** 2)
    return energy, duration


p = Path(__file__).parent
in_path = p.parent / "data/audio/"

pathlist = in_path.rglob("**/*.wav")

empty_files = []
min_energy = 0.01
min_duration = 0.3

for path in pathlist:
    with open(path, "rb") as f:
        energy, duration = energy_duration(f)
        if energy < min_energy or duration < min_duration:
            empty_files.append(str(path))

with open(p / "empty-files.txt", "w") as wf:
    for file in empty_files:
        wf.write(file + "\n")
