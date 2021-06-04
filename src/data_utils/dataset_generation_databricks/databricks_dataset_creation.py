# Databricks notebook source
# MAGIC %md
# MAGIC # Create a Dataset using Databricks
# MAGIC 
# MAGIC Steps
# MAGIC * databricks driver is connected to our postgres DB
# MAGIC * databricks driver reads the scan artifacts from postgres DB
# MAGIC * databricks driver copies all scan artifact blobs to DBFS
# MAGIC * databricks worker can open and process blobs

# COMMAND ----------

! pip install scikit-image
! pip install tqdm
! pip install azure-storage-blob

# COMMAND ----------

from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
from typing import Tuple, List
import zipfile

import numpy as np
import pandas as pd
import psycopg2
from skimage.transform import resize
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient

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
# MAGIC We build our SQL query, so that we get all the required information for ML:
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

# Get a query_result row
if DEBUG:
    query_result_one: Tuple[str] = cur.fetchone()
    file_path = query_result_one[0]; file_path
  
# Get multiple query_result rows
NUM_ARTIFACTS = 3000  # None
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
# MAGIC # Copy mounted files to DBFS
# MAGIC 
# MAGIC In order for databricks to process the blob data, we need to transfer it to the DBFS of the databricks cluster.
# MAGIC 
# MAGIC Note:
# MAGIC * It is not be enough to mount the blob storage. 
# MAGIC * Copying from mount is very very slow.
# MAGIC 
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

if DEBUG:
    file_name = query_results[0][0]; print(file_name)
    with open('asjj.txt', "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

# COMMAND ----------

def download_from_blob_storage(src, dest, container):
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container, blob=file_path)
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
# MAGIC ## Preprocessing utilities
# MAGIC 
# MAGIC In order to preprocess ZIP file to extract a depthmap, we use this code:
# MAGIC [preprocessing.py](https://github.com/Welthungerhilfe/cgm-rg/blob/92efa0febb91c9656ce8e5dbfad953ff7ce721a9/src/utils/preprocessing.py#L12)
# MAGIC 
# MAGIC [file of minor importance](https://github.com/Welthungerhilfe/cgm-ml/blob/c8be9138e025845bedbe7cfc0d131ef668e01d4b/old/cgm_database/command_preprocess.py#L92)

# COMMAND ----------

WIDTH = 240
HEIGHT = 180
NORMALIZATION_VALUE = 7.5
IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH = HEIGHT, WIDTH

def load_depth(fpath: str) -> Tuple[bytes, int, int, float, float]:
    """Take ZIP file and extract depth and metadata
    Args:
        fpath (str): File path to the ZIP
    Returns:
        depth_data (bytes): depthmap data
        width(int): depthmap width in pixel
        height(int): depthmap height in pixel
        depth_scale(float)
        max_confidence(float)
    """

    with zipfile.ZipFile(fpath) as z:
        with z.open('data') as f:
            # Example for a first_line: '180x135_0.001_7_0.57045287_-0.0057296_0.0022602521_0.82130724_-0.059177425_0.0024800065_0.030834956'
            first_line = f.readline().decode().strip()

            file_header = first_line.split("_")

            # header[0] example: 180x135
            width, height = file_header[0].split("x")
            width, height = int(width), int(height)
            depth_scale = float(file_header[1])
            max_confidence = float(file_header[2])

            depth_data = f.read()
    return depth_data, width, height, depth_scale, max_confidence


def parse_depth(tx: int, ty: int, data: bytes, depth_scale: float, width: int) -> float:
    assert isinstance(tx, int)
    assert isinstance(ty, int)

    depth = data[(ty * width + tx) * 3 + 0] << 8
    depth += data[(ty * width + tx) * 3 + 1]

    depth *= depth_scale
    return depth

def preprocess_depthmap(depthmap):
    return depthmap.astype("float32")

def preprocess(depthmap):
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / NORMALIZATION_VALUE
    depthmap = resize(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    return depthmap
  
def prepare_depthmap(data: bytes, width: int, height: int, depth_scale: float) -> np.array:
    """Convert bytes array into np.array"""
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            # depth data scaled to be visible
            output[cx][height - cy - 1] = parse_depth(cx, cy, data, depth_scale, width)
    arr = np.array(output, dtype='float32')
    return arr.reshape(width, height)

def get_depthmaps(fpaths):
    depthmaps = []
    for fpath in fpaths:
        data, width, height, depthScale, _ = load_depth(fpath)
        depthmap = prepare_depthmap(data, width, height, depthScale)
        depthmap = preprocess(depthmap)
        depthmaps.append(depthmap)

    depthmaps = np.array(depthmaps)
    return depthmaps

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform utilities

# COMMAND ----------

def create_and_save_pickle(zip_input_full_path, timestamp, scan_id, scan_step, target_tuple, order_number):
    """Side effect: Saves and returns file path"""
    depthmaps = get_depthmaps([zip_input_full_path])
    if DEBUG:
        print(depthmaps.shape, depthmaps[0,0,0,0])
    
    pickle_output_path = f"qrcode/{scan_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"  # '/tmp/abc.p'
    pickle_output_full_path = f"/dbfs{DBFS_DIR}/{pickle_output_path}"
    Path(pickle_output_full_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump((depthmaps, np.array(target_tuple)), open(pickle_output_full_path, "wb"))
    return pickle_output_full_path

# COMMAND ----------

def process_artifact_tuple(artifact_tuple):
    """Side effect: Saves and returns file path"""
    artifact_dict = {idx2col[i]: el for i, el in enumerate(artifact_tuple)}
    if DEBUG:
        print('artifact_dict', artifact_dict)
    target_tuple = (artifact_dict['height'], artifact_dict['weight'], artifact_dict['muac'])
    zip_input_full_path = f"/dbfs{DBFS_DIR}/{artifact_dict['file_path']}"

    pickle_output_full_path = create_and_save_pickle(
        zip_input_full_path=zip_input_full_path,
        timestamp=artifact_dict['timestamp'],
        scan_id=artifact_dict['scan_id'],
        scan_step=artifact_dict['scan_step'],
        target_tuple=target_tuple,
        order_number=artifact_dict['order_number'],
    )
    return pickle_output_full_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform

# COMMAND ----------

if DEBUG:
  # Process on the spark driver
  for query_result in query_results[:2]:
      _ = process_artifact_tuple(query_result) 

# COMMAND ----------

rdd = spark.sparkContext.parallelize(query_results, 2)
if DEBUG:
    print(rdd.top(3))  # Inspect first items
    print(rdd.count())

# COMMAND ----------

rdd_processed = rdd.map(process_artifact_tuple)
processed_fnames = rdd_processed.collect()  # processed_fnames = rdd_processed.take(12)
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

def remove_prefix(text, prefix):
    if text.startswith(prefix): return text[len(prefix):]
    return text

prefix = f"/dbfs{DBFS_DIR}/"

def upload_to_blob_storage(src, dest, container, directory):
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container, blob=os.path.join(directory,dest))
    # print(f"Uploading to Azure Storage as blob: {directory}/{dest}")
    with open(src, "rb") as data:
        blob_client.upload_blob(data, overwrite=False)

# COMMAND ----------

if DEBUG:
    print(len(processed_fnames))
    processed_fnames[0]
    full_name = processed_fnames[0]
    assert prefix in full_name
    fname = remove_prefix(full_name, prefix)
    upload_to_blob_storage(src=full_name, dest=fname, container=CONTAINER_NAME_DATASET)

# COMMAND ----------

DATASET_NAME = "dataset"
directory = datetime.now(timezone.utc).strftime(f"{DATASET_NAME}-%Y-%m-%d-%H-%M-%S")
for full_name in tqdm(processed_fnames):
    assert prefix in full_name
    fname = remove_prefix(full_name, prefix)
    upload_to_blob_storage(src=full_name, dest=fname, container=CONTAINER_NAME_DATASET, directory=directory)

# COMMAND ----------


