from constant import REPO_DIR, WHO_tables
from table import Table, ReformatTable
from Observation import Value
from calculate import Zscore
from decimal import Decimal as D
import json


TABLE_REPO = REPO_DIR / 'tables'

test = {}
for table in WHO_tables:
    value = Table(TABLE_REPO / table).loadTable()
    table_name, underscore, zscore_part = table.split('.')[0].rpartition('_')
    reformatTable = ReformatTable(value)
    new_value = reformatTable.add_value()
    new_dict = reformatTable.append_value(new_value, new_value['field_name'])
    test[table_name] = new_dict


def z_score_calculation(table):
    table_name = table.resolve_table()
    value = table.get_values(table_name, test)
    skew, median, coff, y = table.resolve_value(value)
    zscore = Zscore(skew, median, coff, y)
    return zscore.z_score_measurement()


def zScore_wfa(weight=None, muac=None, age_in_days=None, sex=None, height=None):
    table = Value('wfa', weight=weight, muac=None, age_in_days=age_in_days, sex=sex, height=None)
    return z_score_calculation(table)


def zScore_wfh(weight=None, muac=None, age_in_days=None, sex=None, height=None):
    if D(age_in_days) <= 731:
        return zScore_wfl(weight, muac, age_in_days, sex, height)
    table = Value('wfh', weight=weight, muac=None, age_in_days=age_in_days, sex=sex, height=height)
    return z_score_calculation(table)


def zScore_wfl(weight=None, muac=None, age_in_days=None, sex=None, height=None):
    if D(age_in_days) > 731:
        return zScore_wfh(weight, muac, age_in_days, sex, height)
    table = Value('wfl', weight=weight, muac=None, age_in_days=age_in_days, sex=sex, height=height)
    return z_score_calculation(table)


def zScore_lhfa(weight=None, muac=None, age_in_days=None, sex=None, height=None):
    table = Value('lhfa', weight=None, muac=None, age_in_days=age_in_days, sex=sex, height=height)
    return z_score_calculation(table)


def zScore_withclass(weight=None, muac=None, age_in_days=None, sex=None, height=None):
    wfa = zScore_wfa(
        weight=weight, age_in_days=age_in_days, sex=sex)
    if wfa < -3:
        class_wfa = 'Severely Under-weight'
    elif wfa >= -3 and wfa < -2:
        class_wfa = 'Moderately Under-weight'
    else:
        class_wfa = 'Healthy'

    if D(age_in_days) > 24:
        dummy = 'Z_score_WFH'
        dummy_class = "Class_WFH"
        wfl = zScore_wfh(
            weight=weight, age_in_days=age_in_days, sex=sex, height=height)
    else:
        dummy = 'Z_score_WFL'
        dummy_class = "Class_WFL"
        wfl = zScore_wfl(
            weight=weight, age_in_days=age_in_days, sex=sex, height=height)
    class_wfl = SAM_MAM(weight, muac, age_in_days, sex, height)

    lhfa = zScore_lhfa(
        age_in_days=age_in_days, sex=sex, height=height)
    if lhfa < -3:
        class_lhfa = 'Severely Stunted'
    elif lhfa >= -3 and lhfa < -2:
        class_lhfa = 'Moderately Stunted'
    else:
        class_lhfa = 'Healthy'

    zscore = json.dumps({'Z_score_WFA': wfa, 'Class_WFA': class_wfa, dummy: wfl,
                         dummy_class: class_wfl, 'Z_score_HFA': lhfa, 'Class_HFA': class_lhfa})
    return zscore


def SAM_MAM(weight=None, muac=None, age_in_days=None, sex=None, height=None):
    assert muac is not None

    if D(age_in_days) > 731:
        wfl = zScore_wfh(
            weight=weight, age_in_days=age_in_days, sex=sex, height=height)
    else:
        wfl = zScore_wfl(
            weight=weight, age_in_days=age_in_days, sex=sex, height=height)
    if wfl < -3 or D(muac) < 11.5:
        return "SAM"
    elif (wfl >= -3 and wfl < -2) or D(muac) < 12.5:
        return "MAM"
    else:
        return "Healthy"
