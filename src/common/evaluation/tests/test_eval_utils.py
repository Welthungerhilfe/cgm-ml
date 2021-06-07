import sys
from pathlib import Path

from bunch import Bunch
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[2]))  # common/ dir

from evaluation.constants_eval import COLUMN_NAME_AGE, COLUMN_NAME_SEX  # noqa: E402
from evaluation.eval_utils import avgerror, calculate_performance, extract_scantype, extract_qrcode  # noqa: E402
from evaluation.eval_utilities import (  # noqa: E402
    calculate_accuracies, calculate_accuracies_on_age_buckets, calculate_performance_age)

QR_CODE_1 = "1585013006-yqwb95138e"
QR_CODE_2 = "1555555555-yqqqqqqqqq"

RESULT_CONFIG = Bunch(dict(
    ACCURACIES=[.2, .4, .6, 1., 1.2, 2., 2.5, 3., 4., 5., 6.],
    AGE_BUCKETS=[0, 1, 2, 3, 4, 5],
    ACCURACY_MAIN_THRESH=0.5,
))


def prepare_df(df):
    df['scantype'] = df.apply(extract_scantype, axis=1)
    df['qrcode'] = df.apply(extract_qrcode, axis=1)
    df = df.groupby(['qrcode', 'scantype']).mean()
    df['error'] = df.apply(avgerror, axis=1)
    return df


def test_calculate_accuracies():
    data = {
        COLUMN_NAME_SEX: [0., 1., 1.],  # one female and two males
        'error': [0.1, 1.0, 0.0],
    }
    df = pd.DataFrame.from_dict(data)
    accuracy_list = calculate_accuracies([0., 1.], df, COLUMN_NAME_SEX, accuracy_thresh=0.5)
    assert accuracy_list == [100., 50.]

    accuracy_list = calculate_accuracies([0., 1.], df, COLUMN_NAME_SEX, accuracy_thresh=1.1)
    assert accuracy_list == [100., 100.]

    accuracy_list = calculate_accuracies([0., 1.], df, COLUMN_NAME_SEX, accuracy_thresh=0.05)
    assert accuracy_list == [0., 50.]


def test_calculate_accuracies_on_age_buckets():
    data = {
        # one less than 1 year, two 2-year-old, one 3-year-old
        COLUMN_NAME_AGE: [int(365 * 0.5), int(365 * 2.5), int(365 * 2.6), int(365 * 3)],
        'error': [1.2, 1.1, 0.4, 0.1],
    }
    df = pd.DataFrame.from_dict(data)
    age_thresholds_in_years = [0, 1, 2, 3, 4, 5]
    age_buckets = list(zip(age_thresholds_in_years[:-1], age_thresholds_in_years[1:]))

    accuracy_list = calculate_accuracies_on_age_buckets(age_buckets, df, COLUMN_NAME_AGE, accuracy_thresh=0.5)
    assert accuracy_list == [0., 0., 50., 100., 0.]


def test_calculate_performance_age():
    data = {
        'artifacts': [
            f'scans/{QR_CODE_1}/100/pc_{QR_CODE_1}_1591849321035_100_000.p',
            f'scans/{QR_CODE_2}/100/pc_{QR_CODE_2}_1591849321035_100_000.p'],
        COLUMN_NAME_AGE: [int(365 * 0.5), int(365 * 3)],
        'GT': [98.1, 98.9],
        'predicted': [98.1, 98.9],
    }
    df = pd.DataFrame.from_dict(data)
    df = prepare_df(df)
    df_out = calculate_performance_age(code='100', df_mae=df, result_config=RESULT_CONFIG)
    assert len(df_out == 1)
    assert np.all(df_out['0 to 1'] == 100)
    list(df_out.loc[0]) == [100., 0., 0., 100., 0.]


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
    df_out = calculate_performance(code='100', df_mae=df, result_config=RESULT_CONFIG)
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
    df_out = calculate_performance(code='100', df_mae=df, result_config=RESULT_CONFIG)
    assert (df_out[1.2] == 50.0).all()
