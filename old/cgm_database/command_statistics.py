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

main_connector = dbutils.connect_to_main_database()

def execute_command_statistics():
    result_string = ""
    
    # Getting table sizes.
    tables = ["person", "measure", "artifact", "artifact_quality"]
    for table in tables:
        sql_statement = "SELECT COUNT(*) FROM {};".format(table)
        result = main_connector.execute(sql_statement, fetch_one=True)[0]
        result_string += "Table '{}' has {} entries.\n".format(table, result)
    
    # Find the number of rows that lack measurement-id.
    sql_statement = "SELECT COUNT(*) FROM artifact WHERE measure_id IS NULL;"
    result = main_connector.execute(sql_statement, fetch_one=True)[0]
    result_string += "Table 'artifact' has {} entries without measure-id.\n".format(result)

    artifact_types = ["pcd", "rgb"]
    for artifact_type in artifact_types:
        sql_statement = "SELECT COUNT(*) FROM artifact WHERE type='{}';".format(artifact_type)
        result = main_connector.execute(sql_statement, fetch_one=True)[0]
        result_string += "Table 'artifact' has {} entries with type '{}'.\n".format(result, artifact_type)


    print(result_string)

    
if __name__ == "__main__":
    execute_command_statistics()
   