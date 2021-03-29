from .model_utils_constants import BLACKLIST_QRCODES


def preprocess_depthmap(depthmap):
    return depthmap.astype("float32")


def preprocess_targets(targets, targets_indices):
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")


def filter_blacklisted_qrcodes(qrcode_paths):
    qrcode_paths_filtered = []
    for qrcode_path in qrcode_paths:
        qrcode_str = qrcode_path.split('/')[-1]
        assert '-' in qrcode_str and len(qrcode_str) == 21, qrcode_str
        if qrcode_str in BLACKLIST_QRCODES:
            continue
        qrcode_paths_filtered.append(qrcode_path)
    return qrcode_paths_filtered
