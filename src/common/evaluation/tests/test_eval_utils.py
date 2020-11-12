import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parents[2]))  # common/ dir

from evaluation.eval_utils import avgerror, calculate_performance, extract_scantype, extract_qrcode  # noqa: E402


QR_CODE_1 = "1585013006-yqwb95138e"
QR_CODE_2 = "1555555555-yqqqqqqqqq"


def prepare_df(df):
    df['scantype'] = df.apply(extract_scantype, axis=1)
    df['qrcode'] = df.apply(extract_qrcode, axis=1)
    df = df.groupby(['qrcode', 'scantype']).mean()
    df['error'] = df.apply(avgerror, axis=1)
    return df


def test_calculate_performance_100percent():
    data = {
        'artifacts': [
            f'scans/{QR_CODE_1}/100/pc_{QR_CODE_1}_1591849321035_100_000.p',
            f'scans/{QR_CODE_2}/100/pc_{QR_CODE_2}_1591849321035_100_000.p'],
        'GT': [98.1, 98.9],
        'predicted': [98.1, 98.9],
    }
    df = pd.DataFrame.from_dict(data)
    df = prepare_df(df)
    df_out = calculate_performance(code='100', df_mae=df)
    assert (df_out[1.2] == 100.0).all()


def test_calculate_performance_50percent():
    data = {
        'artifacts': [
            f'scans/{QR_CODE_1}/100/pc_{QR_CODE_1}_1591849321035_100_000.p',
            f'scans/{QR_CODE_2}/100/pc_{QR_CODE_2}_1591849321035_100_000.p'],
        'GT': [98.1, 98.9],
        'predicted': [98.1, 98.9 + 7],
    }
    df = pd.DataFrame.from_dict(data)
    df = prepare_df(df)
    df_out = calculate_performance(code='100', df_mae=df)
    assert (df_out[1.2] == 50.0).all()
