import pickle as pk


def pickle(data, file_path):
    with open(file_path, "wb") as f:
        pk.dump(data, f, pk.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, "rb") as f:
        data = pk.load(f)
    return data
