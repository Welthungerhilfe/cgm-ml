import azureml
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.run import Run
import os
import glob2 as glob
import numpy as np
import random
from preprocessing import preprocess_pointcloud, preprocess_targets

# Get the current run.
run = Run.get_context()

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    print("Running in offline mode...")
    
    # Access workspace.
    print("Accessing workspace...")
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "gapnet-offline")
    run = experiment.start_logging(outputs=None, snapshot_directory=".")
    
    # Get dataset.
    print("Accessing dataset...")
    if os.path.exists("premiumfileshare") == False:
        dataset_name = "cgmmldevpremium-SampleDataset-Example"
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path='.', overwrite=False)
    dataset_path = glob.glob(os.path.join("premiumfileshare", "*"))[0]

# Online run. Use dataset provided by training notebook.
else:
    print("Running in online mode...")
    experiment = run.experiment
    workspace = experiment.workspace
    dataset_path = run.input_datasets["dataset"]


# The labeled dataset specific part. FileHandlingOption would also allow the .MOUNT option
import azureml.contrib.dataset
from azureml.contrib.dataset import FileHandlingOption
data_pd = dataset_path.to_pandas_dataframe(file_handling_option=FileHandlingOption.DOWNLOAD, target_path='./download/', overwrite_download=True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# The labeled dataset has a image_url column which references the image to the label
#read images from downloaded path
img = mpimg.imread(data_pd.loc[0,'image_url'])
print("Successfully loaded the image")
print(len(data_pd))

# +

run.complete()