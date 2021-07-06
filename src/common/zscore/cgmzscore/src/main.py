from constant import REPO_DIR, WHO_tables
from table import Table, ReformatTable
from Observation import Value
from decimal import Decimal as D
from calculate import Zscore


TABLE_REPO = REPO_DIR / 'tables'


if __name__ == "__main__":
    test = {}
    for table in WHO_tables:
        value = Table(TABLE_REPO / table).loadTable()
        table_name, underscore, zscore_part = table.split('.')[0].rpartition('_')
        reformatTable = ReformatTable(value)
        new_value = reformatTable.add_value()
        new_dict = reformatTable.append_value(new_value, new_value['field_name'])
        test[table_name] = new_dict
    chart = 'wfh'
    weight = "7.853"
    muac = "13.5"
    age_in_days = "16"
    sex = "M"
    height = "73"
    table = Value(chart='wfh', weight="7.853", muac="13.5", age_in_days="16",
                  sex="M", height="73")
    table_name = table.resolve_table()
    value = table.get_values(table_name, test)
    skew, median, coff, y = table.resolve_value(value)
    zscore = Zscore(skew, median, coff, y)
    print(zscore.z_score_measurement())
