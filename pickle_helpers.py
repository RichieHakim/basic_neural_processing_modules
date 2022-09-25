import pickle
from pathlib import Path

def simple_save(obj, filename, mode='wb', mkdir=False):
    if mkdir:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, mode) as f:
        pickle.dump(obj, f)


def simple_load(filename, mode='rb'):
    with open(filename, mode) as f:
        return pickle.load(f)


def save_dict(obj, filename, mode='wb'):
    with open(filename, mode) as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
