# https://stackoverflow.com/questions/42942705/sorting-file-based-on-filename-match-with-folder-python

from pathlib import Path
import shutil, os


p = Path(__file__).parent
in_path = p / "images"
out_path = p / "data"

pathlist = in_path.rglob("**/*.png")

count = 0
for i, path in enumerate(pathlist):
    train_path = f"{out_path}/train/{path.parent.name}"
    test_path = f"{out_path}/test/{path.parent.name}"
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if count == 10:
        shutil.move(str(path), str(test_path))
        count = 0
    else:
        shutil.move(str(path), str(train_path))
        count += 1
