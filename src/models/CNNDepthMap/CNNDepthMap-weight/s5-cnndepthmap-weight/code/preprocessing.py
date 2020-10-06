def preprocess_depthmap(depthmap):
    # TODO here be more code.
    return depthmap.astype("float32")


def preprocess_targets(targets, targets_indices):
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")
