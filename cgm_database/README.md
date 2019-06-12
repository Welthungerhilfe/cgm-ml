# Database.

This folder contains means and methods to access the CGM database. 

## Things you need to know.

- There is also a schema for our database (schema.sql). It creates all our necessary tables.
- Every user has to create their own credentials file (dbconnection.json). This is deliberately not maintained via git for security reasons. The file is supposed to look like this:
```
{
    "dbname": "cgm_ml_db",
    "user": INSERT_USER_NAME_HERE,
    "host": "cgm-dev.postgres.database.azure.com",
    "password": INSERT_PASSWORD-ERE,
    "port": 5432,
    "sslmode": "require"
}
```
- There are a couple of notebooks available that will show you basic functionality.

## Available commands.

- command_init_database.py: Initializes the database using schema.sql.
- command_update_measurements.py: Synchronizes the measurements table.
- command_update_media.py: Updates the media tables.
- command_statistics.py: Prints database statistics.
- command_preprocess.py: Reads from the database and writes data preprocessed for training.
- command_update_artifactsquality.py: Determines the quality of artifacts with respect to a trained model.
