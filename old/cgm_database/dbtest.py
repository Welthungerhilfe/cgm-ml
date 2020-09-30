import dbutils
import datetime

table = "image_data"

main_connector = dbutils.connect_to_main_database()

#main_connector.clear_table(table)

sql_statement = "SELECT qrcode FROM measurements WHERE qrcode != 'NaN';"
results = main_connector.execute(sql_statement, fetch_all=True)
results = [result[0] for result in results]
print(sorted(list(set(results))))


exit(0)

insert_data = {}
insert_data["path"] = "somepath"
insert_data["qrcode"] = "someqrcode"
insert_data["targets"] = "10, 20" 
insert_data["last_updated"] = str(datetime.datetime.now())
insert_data["rejected_by_expert"] = False
insert_data["had_error"] = False
insert_data["error_message"] = ""
insert_data["width_px"] = 128
insert_data["height_px"] = 127
insert_data["blur_variance"] = 1.0
sql_statement = dbutils.create_insert_statement(table, insert_data.keys(), insert_data.values())
print(sql_statement)
results = main_connector.execute(sql_statement)
print(results)

sql_statement = dbutils.create_select_statement(table, ["path"], ["somepath"])
print(sql_statement)
results = main_connector.execute(sql_statement, fetch_all=True)
print(len(results))
