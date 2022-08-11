
def features_by_channels(epochs, axis=1):
    X = []
    for c in range(epochs.shape[axis]):
        X.append(epochs[:,c,:])
    return X