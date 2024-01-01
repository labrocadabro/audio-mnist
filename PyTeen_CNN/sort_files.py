# https://stackoverflow.com/questions/42942705/sorting-file-based-on-filename-match-with-folder-python

from pathlib import Path
import shutil, os


def sort_files(dir_name):
    p = Path(__file__).parent
    in_path = p / dir_name
    out_path = p / "processed"
    for i in range(10):
        digit_folder = f"{out_path}/{i}"
        if not os.path.exists(digit_folder):
            os.makedirs(digit_folder)

    pathlist = in_path.rglob("**/*.wav")

    for i, path in enumerate(pathlist):
        print(path)
        shutil.move(str(path), f"{out_path}/{path.stem[0]}")


sort_files("data/original")
