import pickle
from pathlib import Path

def simple_save(obj, filename, mkdir=False):
    if mkdir:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def simple_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_dict(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
