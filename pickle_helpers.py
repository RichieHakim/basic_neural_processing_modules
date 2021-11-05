import pickle

def simple_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def simple_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_dict(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
