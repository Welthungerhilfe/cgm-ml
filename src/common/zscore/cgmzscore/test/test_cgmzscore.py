'''
# Test case from data provide in survey of https://www.who.int/childgrowth/software/
'''
import os
import logging
import logging.config

import pandas as pd
from pathlib import Path

import sys
print(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[1] / 'src'))

from main import zScore_wfa, zScore_lhfa, zScore_wfh, zScore_wfl  # noqa


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

module_dir = str(os.path.split(os.path.abspath(__file__))[0])


df = pd.read_csv(module_dir + '/testdata.csv')

# 1 for male
# 2 for female


def check_null(i, df):
    return (pd.isnull(df.loc[i, 'WEIGHT'])
            or pd.isnull(df.loc[i, '_agedays'])
            or pd.isnull(df.loc[i, '_ZWEI'])
            or df.loc[i, '_agedays'] == 0
            or pd.isnull(df.loc[i, '_ZLEN'])
            or pd.isnull(df.loc[i, '_ZWFL']))


def test_zScore_wfa():
    for i in range(len(df)):
        if check_null(i, df):
            continue

        sex = 'M' if df['GENDER'][i] == 1 else 'F'

        g = float("{0:.9f}". format(df['WEIGHT'][i]))
        v = zScore_wfa(weight=str(g), height=str(
            df['HEIGHT'][i]), sex=sex, age_in_days=str((df['_agedays'][i])))
        ans = float("{0:.2f}". format(abs(v - df['_ZWEI'][i])))
        assert ans <= 0.01


test_zScore_wfa()


def test_zScore_lhfa():
    for i in range(len(df)):
        if check_null(i, df):
            continue

        sex = 'M' if df['GENDER'][i] == 1 else 'F'

        g = float("{0:.9f}". format(df['HEIGHT'][i]))

        v = zScore_lhfa(height=str(g), sex=sex,
                        age_in_days=str((df['_agedays'][i])))

        logging.info(g)

        ans = float("{0:.2f}". format(abs(v - df['_ZLEN'][i])))
        assert ans <= 0.01


def test_zScore_wfh():
    for i in range(len(df)):
        if check_null(i, df):
            continue

        sex = 'M' if df['GENDER'][i] == 1 else 'F'

        g = float("{0:.5f}". format(df['HEIGHT'][i]))
        t = float("{0:.9f}". format(df['WEIGHT'][i]))

        v = zScore_wfh(height=str(g), weight=str(
            t), sex=sex, age_in_days=str((df['_agedays'][i])))

        ans = float("{0:.2f}". format(abs(v - df['_ZWFL'][i])))
        assert ans <= 0.01


def test_zScore_wfl():
    for i in range(len(df)):
        if check_null(i, df):
            continue

        sex = 'M' if df['GENDER'][i] == 1 else 'F'

        g = float("{0:.5f}". format(df['HEIGHT'][i]))
        t = float("{0:.9f}". format(df['WEIGHT'][i]))

        v = zScore_wfl(height=str(g), weight=str(
            t), sex=sex, age_in_days=str((df['_agedays'][i])))

        ans = float("{0:.2f}". format(abs(v - df['_ZWFL'][i])))
        assert ans <= 0.01
