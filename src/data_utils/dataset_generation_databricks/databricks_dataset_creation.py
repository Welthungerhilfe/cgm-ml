# Databricks notebook source
# MAGIC %md
# MAGIC # Create a Dataset using Databricks
# MAGIC
# MAGIC Steps
# MAGIC * databricks driver gets list of artifacts from postgres DB
# MAGIC * databricks driver copies artifacts (from blob stoarage to DBFS)
# MAGIC * databricks workers process artifacts
# MAGIC * databricks driver uploads all the blobs (from DBFS to blob storage)

# COMMAND ----------

# flake8: noqa

! pip install scikit-image
! pip install tqdm
! pip install azure-storage-blob

! pip install --upgrade cgm-ml-common

# COMMAND ----------

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import psycopg2
from skimage.transform import resize
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient

from src.common.data_utilities.mlpipeline_utils import ArtifactProcessor

# COMMAND ----------

# Constants
ENV_PROD = "env_prod"
ENV_SANDBOX = "env_sandbox"

# COMMAND ----------

# Configuration
DEBUG = False
ENV = ENV_SANDBOX

MOUNT_POINT = f"/mnt/{ENV}_input"
MOUNT_DATASET = f"/mnt/{ENV}_dataset"
DBFS_DIR = f"/tmp/{ENV}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Access SQL database to find all the scans/artifacts of interest
# MAGIC
# MAGIC #### SQL query
# MAGIC
# MAGIC We build our SQL query, so that we get all the required information for the ML dataset creation:
# MAGIC - the artifacts (depthmap, RGB, pointcloud)
# MAGIC - the targets (measured height, weight, and MUAC)
# MAGIC
# MAGIC The ETL packet shows which tables are involved
# MAGIC
# MAGIC ![image info](https://dev.azure.com/cgmorg/e5b67bad-b36b-4475-bdd7-0cf6875414df/_apis/git/repositories/465970a9-a8a5-4223-81c1-2d3f3bd4ab26/Items?path=%2F.attachments%2Fcgm-solution-architecture-etl-draft-ETL-samplling-71a42e64-72c4-4360-a741-1cfa24622dce.png&download=false&resolveLfs=true&%24format=octetStream&api-version=5.0-preview.1&sanitize=true&versionDescriptor.version=wikiMaster)
# MAGIC
# MAGIC The query will produce one artifact per row.

# COMMAND ----------

SECRET_SCOPE = "cgm-ml-ci-dev-databricks-secret-scope"
if ENV == ENV_SANDBOX:
    host = "cgm-ml-ci-dev-mlapi-psql.postgres.database.azure.com"
    user = dbutils.secrets.get(scope=SECRET_SCOPE, key="psql-username")
    password = dbutils.secrets.get(scope=SECRET_SCOPE, key="psql-password")
elif ENV == ENV_PROD:
    host = "cgm-ml-ci-prod-mlapi-psql.postgres.database.azure.com"
    user = dbutils.secrets.get(scope=SECRET_SCOPE, key="prod-psql-username")
    password = dbutils.secrets.get(scope=SECRET_SCOPE, key="prod-psql-password")
else:
    raise Exception(f"Unknown environment: {ENV}")

conn = psycopg2.connect(host=host, database='cgm-ml', user=user, password=password)

# COMMAND ----------

cur = conn.cursor()

# COMMAND ----------

SQL_QUERY = """
SELECT f.file_path, f.created as timestamp,
       s.id as scan_id, s.scan_type_id as scan_step,
       m.height, m.weight, m.muac,
       a.ord as order_number
FROM file f
INNER JOIN artifact a ON f.id = a.file_id
INNER JOIN scan s     ON s.id = a.scan_id
INNER JOIN measure m  ON m.person_id = s.person_id
WHERE a.format = 'depth'
"""
cur.execute(SQL_QUERY)

# Get multiple query_result rows
NUM_ARTIFACTS = 30  # None
query_results: List[Tuple[str]] = cur.fetchall() if NUM_ARTIFACTS is None else cur.fetchmany(NUM_ARTIFACTS)

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation of a file_path**
# MAGIC
# MAGIC The SQL result provides file_paths which have this format
# MAGIC
# MAGIC ```
# MAGIC Example: '1618896404960/2fe0ee0e-daf0-45a4-931e-cfc7682e1ce6'
# MAGIC Format: f'{unix-timeatamp}/{random uuid}'
# MAGIC ```

# COMMAND ----------

df = pd.DataFrame(query_results, columns=list(map(lambda x: x.name, cur.description)))
print(df.shape)
df.head()

# COMMAND ----------

col2idx = {col.name: i for i, col in enumerate(cur.description)}; print(col2idx)
idx2col = {i: col.name for i, col in enumerate(cur.description)}; print(idx2col)

# COMMAND ----------

# MAGIC %md
# MAGIC # Copy artifact files to DBFS
# MAGIC
# MAGIC In order for databricks to process the blob data, we need to transfer it to the DBFS of the databricks cluster.
# MAGIC
# MAGIC Note:
# MAGIC * Copying from mount is very very slow, therefore we copy the data
# MAGIC
# MAGIC ## Download blobs
# MAGIC
# MAGIC We use [Manage blobs Python SDK](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#download-blobs)
# MAGIC to download blobs directly from the Storage Account(SA) to [DBFS](https://docs.databricks.com/data/databricks-file-system.html).

# COMMAND ----------

if ENV == ENV_SANDBOX:
    CONNECTION_STR = dbutils.secrets.get(scope=SECRET_SCOPE, key="dev-sa-connection-string")
    STORAGE_ACCOUNT_NAME = "cgmmlcidevmlapisa"
    CONTAINER_NAME = "cgm-result"
elif ENV == ENV_PROD:
    raise Exception("Not yet setup this SA connection string")
else:
    raise Exception(f"Unknown environment: {ENV}")

# COMMAND ----------

BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(CONNECTION_STR)

# COMMAND ----------

def download_from_blob_storage(src: str, dest: str, container: str):
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container, blob=src)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as download_file:
        content = blob_client.download_blob().readall()
        download_file.write(content)

# COMMAND ----------

file_path_idx = col2idx['file_path']
for res in tqdm(query_results):
    file_path = res[file_path_idx]
    download_from_blob_storage(src=file_path, dest=f"dbfs:{DBFS_DIR}/{file_path}", container=CONTAINER_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform ZIP into pickle
# MAGIC
# MAGIC Here we document the format of the artifact path
# MAGIC
# MAGIC ```
# MAGIC f"scans/1583462505-43bak4gvfa/101/pc_1583462505-43bak4gvfa_1591122173510_101_002.p"
# MAGIC f"qrcode/{scan_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"
# MAGIC ```
# MAGIC
# MAGIC Idea for a future format could be to include person_id like so:
# MAGIC ```
# MAGIC f"qrcode/{person_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform

# COMMAND ----------

rdd = spark.sparkContext.parallelize(query_results, 2)

# COMMAND ----------

input_dir = f"/dbfs{DBFS_DIR}"
output_dir = f"/dbfs{DBFS_DIR}"
artifact_processor = ArtifactProcessor(input_dir, output_dir, idx2col)

# COMMAND ----------

rdd_processed = rdd.map(artifact_processor.process_artifact_tuple)
processed_fnames = rdd_processed.collect()
print(processed_fnames[:3])

# COMMAND ----------

# MAGIC %md
# MAGIC # Upload to blob storage
# MAGIC
# MAGIC We use [Manage blobs Python SDK](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#upload-blobs-to-a-container)
# MAGIC to upload blobs

# COMMAND ----------

if ENV == ENV_SANDBOX:
    STORAGE_ACCOUNT_NAME = "cgmmlcidevmlapisa"
    CONTAINER_NAME_DATASET = "cgm-datasets"
    CONNECT_STR_DATASET = CONNECTION_STR
elif ENV == ENV_PROD:
    raise Exception("Not yet setup this SA connection string")
else:
    raise Exception(f"Unknown environment: {ENV}")

# COMMAND ----------

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix): return text[len(prefix):]
    return text

PREFIX = f"/dbfs{DBFS_DIR}/"

def upload_to_blob_storage(src: str, dest: str, container: str, directory: str):
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container, blob=os.path.join(directory,dest))
    with open(src, "rb") as data:
        blob_client.upload_blob(data, overwrite=False)

# COMMAND ----------

DATASET_NAME = "dataset"
directory = datetime.now(timezone.utc).strftime(f"{DATASET_NAME}-%Y-%m-%d-%H-%M-%S")
for full_name in tqdm(processed_fnames):
    assert PREFIX in full_name
    fname = remove_prefix(full_name, PREFIX)
    upload_to_blob_storage(src=full_name, dest=fname, container=CONTAINER_NAME_DATASET, directory=directory)

# COMMAND ----------


