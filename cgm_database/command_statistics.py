import warnings
warnings.filterwarnings("ignore")
import dbutils

main_connector = dbutils.connect_to_main_database()


def execute_command_statistics():
    result_string = ""
    
    # Getting table sizes.
    tables = ["measurements", "image_data", "pointcloud_data"]
    for table in tables:
        sql_statement = "SELECT COUNT(*) FROM {};".format(table)
        result = main_connector.execute(sql_statement, fetch_one=True)[0]
        result_string += "Table {} has {} entries.\n".format(table, result)
    
    # Find the number of rows that lack measurement-id.
    tables = ["image_data", "pointcloud_data"]
    for table in tables:
        sql_statement = "SELECT COUNT(*) FROM {} WHERE measurement_id IS NULL;".format(table)
        result = main_connector.execute(sql_statement, fetch_one=True)[0]
        result_string += "Table {} has {} entries without measurement-id.\n".format(table, result)
    
    print(result_string)

    
if __name__ == "__main__":
    execute_command_statistics()
   