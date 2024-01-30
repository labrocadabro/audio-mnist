import shutil
from pathlib import Path

p = Path(__file__).parent.parent

empty_list = p / "empty-files.txt"
move_to = p / "data/removed"

with open(empty_list) as f:
    paths = f.read().splitlines()
    for path in paths:
        shutil.move(path, str(move_to / path.split("/")[-1]))
