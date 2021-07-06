import json


class Table:
    def __init__(self, table_file: str):
        self.table_file = table_file

    def loadTable(self) -> list:
        open_table = open(self.table_file, 'r')
        table_value = json.load(open_table)
        return table_value


class ReformatTable:
    def __init__(self, table_value: json):
        self.table_value = table_value

    def add_value(self):
        if 'Length' in self.table_value[0]:
            field_name = 'Length'
        elif 'Height' in self.table_value[0]:
            field_name = 'Height'
        elif 'Day' in self.table_value[0]:
            field_name = 'Day'
        else:
            raise Exception('error loading: %s' % self.table_value)
        new_dict = {'field_name': field_name}
        return new_dict

    def append_value(self, new_dict, field_name):
        for d in self.table_value:
            new_dict.update({d[field_name]: d})
        return new_dict
