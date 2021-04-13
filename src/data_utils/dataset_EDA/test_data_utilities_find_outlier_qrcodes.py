import pandas as pd
from data_utilities import find_outlier_qrcodes

# start preparing the dataFrame here

QR_CODE_1 = "1585013006-yqwb95138e"
QR_CODE_2 = "1555555555-yqqqqqqqqq"
QR_CODE_3 = "1212121212-jajajajaja"
QR_CODE_4 = "9922992299-lelelelele"


def prepare_test_df():
    data = {
        'artifacts': [
            f'scans/{QR_CODE_1}/100/pc_{QR_CODE_1}_1591849321035_100_000.p',
            f'scans/{QR_CODE_2}/100/pc_{QR_CODE_2}_1591849321035_100_000.p',
            f'scans/{QR_CODE_3}/100/pc_{QR_CODE_3}_1591849321035_100_000.p',
            f'scans/{QR_CODE_1}/100/pc_{QR_CODE_1}_1591849321035_100_000.p',
            f'scans/{QR_CODE_4}/100/pc_{QR_CODE_4}_1591849321035_100_000.p'],
        'age': [180, 365, 3000, 180, 20],
        'weight': [3.0, 6.0, 32.0, 3.0, 2.7],
        'height': [30.0, 95.0, 102.0, 30.0, 195.0],
        'qrcode': [QR_CODE_1, QR_CODE_2, QR_CODE_3, QR_CODE_1, QR_CODE_4],
    }
    df = pd.DataFrame.from_dict(data)
    return df


# test my find_outlier for agemin, agemax, weightmin, weightmax, heightmin, heightmax

def test_find_outlier_qrcodes_age_min(df):
    qrs = find_outlier_qrcodes(df, 'age', '<365/2')
    return qrs


def test_find_outlier_qrcodes_age_max(df):
    qrs = find_outlier_qrcodes(df, 'age', '>365*6')
    return qrs


def test_find_outlier_qrcodes_weight_min(df):
    qrs = find_outlier_qrcodes(df, 'weight', '<5.0')
    return qrs


def test_find_outlier_qrcodes_weight_max(df):
    qrs = find_outlier_qrcodes(df, 'weight', '>30.0')
    return qrs


def test_find_outlier_qrcodes_height_min(df):
    qrs = find_outlier_qrcodes(df, 'height', '<40.0')
    return qrs


def test_find_outlier_qrcodes_height_max(df):
    qrs = find_outlier_qrcodes(df, 'height', '>150.0')
    return qrs
