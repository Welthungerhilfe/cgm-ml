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

def execute_command_updatemeasures():
    print("Updating measurements...")
    
    main_connector = dbutils.connect_to_main_database()
    
    # Where to get the data.
    csv_path = config.measure_csv_path
    print("Using {}".format(csv_path))

    # Load the data-frame.
    df = pd.read_csv(csv_path)
    
    # List all columns.
    columns = list(df)
    ignored_columns = ["sex", "address", "qrcode", "latitude", "longitude", "personId"]
    columns_mapping = { column: column for column in columns if column not in ignored_columns}
    columns_mapping["personId"] = "person_id"
    columns_mapping["headCircumference"] = "head_circumference"
    columns_mapping["deletedBy"] = "deleted_by"
    columns_mapping["createdBy"] = "created_by"
    
    table = "measure"

    # Number of rows before.
    rows_number = main_connector.get_number_of_rows(table)
    print("Number of rows before: {}".format(rows_number))

    # Insert data in batches.
    batch_size = 1000
    sql_statement = ""
    rows_number_df = len(df.index)
    bar = progressbar.ProgressBar(max_value=rows_number_df)
    for index, row in df.iterrows():
        bar.update(index)

        keys = []
        values = []
        for df_key, db_key in columns_mapping.items():
            keys.append(str(db_key))
            values.append(str(row[df_key]))
            
        # TODO what is this?
        keys.append("date")
        values.append(int(time.time()))
        
        # TODO what is this?
        keys.append("artifact")
        values.append("UNKNOWN")
        
        sql_statement += dbutils.create_insert_statement(table, keys, values)
        
        if index != 0 and ((index % batch_size) == 0 or index == rows_number_df - 1):
            main_connector.execute(sql_statement)
            sql_statement = ""

    bar.finish()

    # Number of rows after sync.
    rows_number = main_connector.get_number_of_rows(table)
    print("Number of rows after: {}".format(rows_number))
    
    
if __name__ == "__main__":
    execute_command_updatemeasures()