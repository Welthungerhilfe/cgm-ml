import warnings
warnings.filterwarnings("ignore")
import dbutils

def execute_command_init():
    print("Initializing DB...")
    main_connector = dbutils.connect_to_main_database()
    main_connector.execute_script_file("schema.sql")
    print("Done.")
    
if __name__ == "__main__":
    execute_command_init()