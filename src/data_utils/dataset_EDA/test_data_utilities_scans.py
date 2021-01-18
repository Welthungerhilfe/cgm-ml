import pandas as pd

from data_utilities import draw_age_distribution, draw_sex_distribution

# Test data: rebuilding the following from code

# 	qrcode	person_id	age	sex	Years
# 0		DQKAiT6cSJkQW2ya_person_1569888000000_Dj66PtJa...	1420	male	4
# 1		lD0OAaaZ0pcD0zRk_person_1570060800000_1AoTKLlT...	1529	female	4
# 2		ZPFkcYGhq15raY3J_person_1573603200000_lfu3HzFC...	945	female	3
# 3		D7tiN7CtUEM9WoRx_person_1571270400000_Q1hKfLbv...	924	male	3
# 4		8SMAKMkY79LovcBU_person_1570838400000_nMpSBNek...	1320	female	4


def _generate_df():
    qrcode = [
        "1584997475-0195z663pl",
        "1584999865-01t0n240ra",
        "1583942091-02migjdla1",
        "1585011777-031jov4jpw",
        "1585003039-032hniw434",
    ]

    age = [
        1420,
        1529,
        945,
        924,
        1320,
    ]

    sex = [
        "male",
        "female",
        "female",
        "male",
        "female",
    ]

    years = [
        4,
        4,
        3,
        3,
        4,
    ]

    scans = pd.DataFrame(list(zip(qrcode, age, sex, years)),
                         columns=['qrcode', 'age', 'sex', 'Years'])
    return scans


def test_draw_sex_distribution():
    scans = _generate_df()
    draw_sex_distribution(scans)


def test_draw_age_distribution():
    scans = _generate_df()
    draw_age_distribution(scans)
