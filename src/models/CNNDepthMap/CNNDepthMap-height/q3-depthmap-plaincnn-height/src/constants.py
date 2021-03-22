from pathlib import Path

REPO_DIR = Path(__file__).parents[6].absolute()
DATA_DIR_ONLINE_RUN = Path("/tmp/data/")
MODEL_CKPT_FILENAME = "best_model.ckpt"

BLACKLIST_QRCODES = [
    "1585000019-syglokl9nx",  # only to test (part of mini)
    "1585366118-qao4zsk0m3",  # in anon-depthmap-95k, child_height = 12.7, scans/1585366118-qao4zsk0m3/102/pc_1585366118-qao4zsk0m3_1593021766372_102_026.p'
    "1585360775-fa64muouel",  # in anon-depthmap-95k, child_height = 7.9, scans/1585360775-fa64muouel/202/pc_1585360775-fa64muouel_1597205960827_202_002.p',
    '1583855791-ldfc59ywg5',  # in anon-depthmap-95k, child_height
    '1583997882-3jqstr1119',  # in anon-depthmap-95k, child_height
    '1584998372-d85ogmqucw',  # in anon-depthmap-95k, child_height
    '1585274424-3oqa4i262a',  # in anon-depthmap-95k, child_height
    '1585010027-xb21f31tvj',  # in anon-depthmap-95k, pixel_value_max = 714286.0, b'scans/1585010027-xb21f31tvj/101/pc_1585010027-xb21f31tvj_1592674994326_101_015.p'
]
