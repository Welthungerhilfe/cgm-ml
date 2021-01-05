import sys
import yaml

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from data_utils import QRCodeCollector  # noqa: E402

PARAMETERS = 'src/data_utils/dataset_generation/parameters.yml'
DB_CONNECTION = 'src/data_utils/dataset_generation/dbconnection.json'

with open(PARAMETERS, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

DATASET = QRCodeCollector(DB_CONNECTION)

QR_CODE_1 = "1597619052-n0p0wb3o77"
QR_CODE_2 = "1555555555-yqqqqqqqqq"
QR_CODE_3 = "1555836480-mvcbmnfbbb"

DATA = {
    'id': [f'Uyx4dG24QFMtAk4v_artifact-scan-pcd_{QR_CODE_1}_version_5.0',
           f'Uyx4dGdcsadawAw_artifact-scan-pcd_{QR_CODE_1}_version_5.0',
           f'JsbdjObdasghncz_artifact-scan-pcd_{QR_CODE_2}_version_5.0',
           f'JsbdjObdakcxvkj_artifact-scan-pcd_{QR_CODE_3}_version_5.0'],
    'storage_path': [
        f'qrcode/{QR_CODE_1}/measure/1591468683124/pc/pc_{QR_CODE_1}_1591468683124_100_001.pcd',
        f'qrcode/{QR_CODE_1}/measure/1591468683124/pc/pc_{QR_CODE_1}_1591468683124_101_001.pcd',
        f'qrcode/{QR_CODE_2}/measure/1591468683124/pc/pc_{QR_CODE_2}_1591468683124_102_001.pcd',
        f'qrcode/{QR_CODE_3}/measure/1591468683124/pc/pc_{QR_CODE_3}_1591468683124_201_001.pcd'],
    'height': [90.9, 90.9, 92.1, 100.2],
    'weight': [15.3, 15.3, 11.3, 18.5],
    'muac': [13.2, 13.2, 16, 15.3],
    'scan_group': ['train', 'train', 'test', np.nan],
    'key': [100, 101, 102, 201],
    'tag': ['good', 'bad', 'good', 'bad'],
    'age': [1223, 1223, 1645, 987],
    'sex': ['male', 'male', 'female', 'female'],
    'qrcode': [QR_CODE_1, QR_CODE_1, QR_CODE_2, QR_CODE_3]

}
DATAFRAME = pd.DataFrame.from_dict(DATA)


class TestListElements():

    def test_dataset_column(self):
        """
        Function to test required columns match with columns from database.
        """
        expected = list(DATAFRAME.columns)
        expected_data = DATASET.get_all_data()
        result = list(expected_data.columns)
        assert set(expected) == set(result)

    def test_scan_group(self):
        """
        Function to test the scan_group fetch from database
        """
        expected_scangroup_data = DATASET.get_scangroup_data(data=DATAFRAME, scangroup='test')
        result = list(expected_scangroup_data['scan_group'])
        assert 'train' not in result

    def test_unique_qrcode(self):
        """
        Function to fetch the unique qrcodes from the dataframe
        """
        unique_qrcodes = DATASET.get_unique_qrcode(DATAFRAME)
        qrcode = unique_qrcodes['qrcode'].tolist()
        assert np.unique(qrcode).size == len(qrcode)
