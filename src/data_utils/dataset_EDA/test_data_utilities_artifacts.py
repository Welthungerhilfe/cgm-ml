import pandas as pd

from data_utilities import extractqrcode

# Test data: rebuilding the following from code

# artifacts	id	storage_path	height	weight	key	tag	qrcode
# 0	pc_1584997475-0195z663pl_1591588126306_100_000...	DQKAiT6cSJkQW2ya_artifact-scan-pcd_15698880000...	qrcode/1584997475-0195z663pl/measure/159158812...	85.7	9.45	100.0	good	1584997475-0195z663pl
# 1	pc_1584997475-0195z663pl_1591588126306_100_001...	DQKAiT6cSJkQW2ya_artifact-scan-pcd_15698880000...	qrcode/1584997475-0195z663pl/measure/159158812...	85.7	9.45	100.0	good	1584997475-0195z663pl
# 2	pc_1584997475-0195z663pl_1591588126306_100_002...	DQKAiT6cSJkQW2ya_artifact-scan-pcd_15698880000...	qrcode/1584997475-0195z663pl/measure/159158812...	85.7	9.45	100.0	good	1584997475-0195z663pl
# 3	pc_1584997475-0195z663pl_1591588126306_100_003...	DQKAiT6cSJkQW2ya_artifact-scan-pcd_15698880000...	qrcode/1584997475-0195z663pl/measure/159158812...	85.7	9.45	100.0	good	1584997475-0195z663pl
# 4	pc_1584997475-0195z663pl_1591588126306_100_004...	DQKAiT6cSJkQW2ya_artifact-scan-pcd_15698880000...	qrcode/1584997475-0195z663pl/measure/159158812...	85.7	9.45	100.0	good	1584997475-0195z663pl



def _generate_df():
    artifacts = [
        "pc_1584997475-0195z663pl_1591588126306_100_000",
        "pc_1584997475-0195z663pl_1591588126306_100_001",
        "pc_1584997475-0195z663pl_1591588126306_100_002",
        "pc_1584997475-0195z663pl_1591588126306_100_003",
        "pc_1584997475-0195z663pl_1591588126306_100_004",
    ]

    storage_path = [
        "qrcode/1584997475-0195z663pl/measure/159158812...",
        "qrcode/1584997475-0195z663pl/measure/159158812...",
        "qrcode/1584997475-0195z663pl/measure/159158812...",
        "qrcode/1584997475-0195z663pl/measure/159158812...",
        "qrcode/1584997475-0195z663pl/measure/159158812...",
    ]

    height = [
        85.7,
        85.7,
        85.7,
        85.7,
        85.7,
    ]

    weight = [
        9.45,
        9.45,
        9.45,
        9.45,
        9.45,
    ]

    key = [
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
    ]

    tag = [
        "good",
        "good",
        "good",
        "good",
        "good",
    ]

    scans = pd.DataFrame(list(zip(artifacts, storage_path, height, weight, key, tag)),
                         columns =['artifacts', 'storage_path', 'height', 'weight', 'key', 'tag'])
    return scans

def test_extractqrcode():
    scans = _generate_df()
    scans['qrcode'] = scans.apply(extractqrcode, axis=1)

if __name__ == "__main__":
    test_extractqrcode()
    aaa = 1
