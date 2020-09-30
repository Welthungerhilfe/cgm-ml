# Training

Here you will find all training scripts.

Things to take into account:

- Please make sure that the instructions for training reside in `runtraining.sh`. This is the entrypoint.
- Please make sure that training artifacts end up in `/whhdata/models`.
- Please make sure that training artifacts always have a date and a time in the filename.
- Please make sure that each training uses TensorBoard. TensorBoard logs should end up in `/whhdata/models/logs`.

## Configuration

adjust config.py to contain paths to important folders for the following steps

## Check for database schema updates

- `cgm_database/schema.sql`
- `cgm_database/schema_migration_2.0_2.1.sql`

## Load legacy data into database without ETL

### Load persons

`command_update_persons.py`

needs `dbconnection.json`

takes csv file and writes content into person table

### Load measures

`command_update_measures.py`

takes csv file and writes content into measure

### Load artifacts

`command_update_artifacts.py`

takes csv file and writes content into measure

## Preprocess

make changes to preprocess select sql in order to train with the artifacts you want

## Train on proper preprocessed path

in cgm_training run for example `train_pointnet_generator.py path` or change `config.py`
