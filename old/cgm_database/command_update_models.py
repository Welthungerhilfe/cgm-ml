import os
import glob
import json
import shutil
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time
import dbutils
import sys

def main():

    models_file = str(sys.argv[1])
    db_connection_file = str(sys.argv[2])
    table_name = str(sys.argv[3])

    if len(sys.argv) != 4:
        print("usage: command_update_models.py models_file db_connection_file table_name")

    main_connector = dbutils.connect_to_main_database(db_connection_file)

    with open(models_file) as json_file:
        json_data = json.load(json_file)

        for data in json_data["models"]:

            check_existing_models = "SELECT id from {};".format(table_name)
            results = main_connector.execute(check_existing_models, fetch_all=True)
            if data["name"] in str(results):
                print("{} already exists in model table".format(data["name"]))
                continue
            value_mapping = {}
            value_mapping["id"] = data["name"]
            value_mapping["name"] = data["name"]
            version = data["name"].split('_')[0]
            value_mapping["version"] = version
            del data["name"]
            value_mapping["json_metadata"] = json.dumps(data)
            keys = []
            values = []
            for key in value_mapping.keys():
                keys.append(key)
                values.append(value_mapping[key])
            sql_statement = dbutils.create_insert_statement(table_name, keys, values, False, True)
            try:
                results = main_connector.execute(sql_statement)
                print("{} successfully added to model table".format(value_mapping["name"]))
            except Exception as error:
                print(error)

    main_connector.cursor.close()
    main_connector.connection.close()

if __name__ == "__main__":
    main()



