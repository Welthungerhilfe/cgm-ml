#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Tristan Behrens <tristan@ai-guru.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import warnings
warnings.filterwarnings("ignore")
import dbutils
import glob2 as glob
import os
import progressbar
import pandas as pd
import sys
import time
import config

def execute_command_persons():
    print("Updating persons...")
    
    main_connector = dbutils.connect_to_main_database()
    
    # Where to get the data.
    csv_path = config.measure_csv_path
    print("Using {}".format(csv_path))

    # Load the data-frame.
    df = pd.read_csv(csv_path)
    
    # List all columns.
    columns = list(df)
    print(columns)
        
    table = "person"

    # Number of rows before.
    rows_number = main_connector.get_number_of_rows(table)
    print("Number of rows before: {}".format(rows_number))

    # Insert data in batches.
    batch_size = 1000
    sql_statement = ""
    rows_number_df = len(df.index)
    bar = progressbar.ProgressBar(max_value=rows_number_df)
    used_ids = []
    for index, row in df.iterrows():
        bar.update(index)
        
        # Make sure there are no duplicates. Local.
        select_sql_statement = "SELECT COUNT(*) FROM person WHERE id='{}'".format(row["personId"])
        result = main_connector.execute(select_sql_statement, fetch_one=True)[0]
        if row["personId"] in used_ids or result != 0:
            #print(row["personId"], "already in DB")
            pass
        else:
        
            # TODO check all of these.
            insert_data = {}
            insert_data["id"] = row["personId"]
            insert_data["name"] = "UNKNOWN" 
            insert_data["surname"] = "UNKNOWN" 
            insert_data["birthday"] = 0   
            insert_data["sex"] = "UNKNOWN" 
            insert_data["guardian"] = "UNKNOWN" 
            insert_data["is_age_estimated"] = False
            insert_data["qr_code"] = row["qrcode"]    
            insert_data["created"] = 0
            insert_data["timestamp"] = row["timestamp"]
            insert_data["created_by"] = row["createdBy"]
            insert_data["deleted"] = row["deleted"]
            insert_data["deleted_by"] = row["deletedBy"]

            sql_statement += dbutils.create_insert_statement(table, insert_data.keys(), insert_data.values())
            #print(sql_statement)
            used_ids.append(row["personId"])
        
        if index != 0 and ((index % batch_size) == 0 or index == rows_number_df - 1) and sql_statement != "":
            main_connector.execute(sql_statement)
            sql_statement = ""

    bar.finish()

    # Number of rows after sync.
    rows_number = main_connector.get_number_of_rows(table)
    print("Number of rows after: {}".format(rows_number))
    
    
if __name__ == "__main__":
    execute_command_persons()