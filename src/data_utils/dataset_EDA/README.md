## Introduction

Here you will find the Exploratory Data Analysis (EDA) notebooks for our various datasets.

## Setup EDA

If you want to run the EDA notebooks, you most likely need a compute instance
because the datasets are quite big.
Start new compute instance in AzuremL and clone `cgm-ml` repo.
Then run:

```bash
cd cgm-ml
cd src/common/evaluation/QA/
conda env update --file environment.yml
conda activate CGM_QA_Pipeline
cd ../../../..
conda install notebook ipykernel
ipython kernel install --user --name CGM_QA_Pipeline
```

In order to mount all datasets at the same time run:

```bash
cd /root/
vim connection.cfg  # get this from a collegue (e.g. Shashank or Gracjan)
cd /mnt
mkdir datasets
blobfuse /mnt/datasets/ --tmp-path=/mnt/tmp -o attr_timeout=240 -o  entry_timeout=240 -o negative_timeout=120 -o allow_other --config-file=/root/connection.cfg
cd datasets
cd realtime_evaluation  # for example
```

In order to mount a single dataset with python code, run:

```python
from azureml.core import Workspace, Dataset
workspace = Workspace.from_config()
dataset_name = "anon-depthmap-95k"
dataset = workspace.datasets[dataset_name]
inputs = [dataset.as_named_input("dataset").as_mount("/mnt/dataset")]
```

### anon-depthmaps-56k

This Exploratory Data Analysis (EDA) notebook provide the data analysis of the [anon-depthmaps-56k](https://ml.azure.com/dataset/anon-depthmaps-56k/latest/details?wsid=/subscriptions/9b82ecea-6780-4b85-8acf-d27d79028f07/resourceGroups/cgm-ml-prod/providers/Microsoft.MachineLearningServices/workspaces/cgm-azureml-prod&tid=006dabd7-456d-465b-a87f-f7d557e319c8). The dataset contains 2106 scans of children in depthmap and pcd format. The files are in pickle format and the 0 index contains the depthmap, index 1 have height and index 2 weight.

### anon-depthmap-testset

This Exploratory Data Analysis (EDA) notebook provide the data analysis of the [anon-depthmap-testset](https://ml.azure.com/dataset/anon-depthmap-testset/latest/details?wsid=/subscriptions/9b82ecea-6780-4b85-8acf-d27d79028f07/resourceGroups/cgm-ml-prod/providers/Microsoft.MachineLearningServices/workspaces/cgm-azureml-prod&tid=006dabd7-456d-465b-a87f-f7d557e319c8). The dataset contains 1745 scans of children in depthmap format. The files are in pickle format and the 0 index contains the depthmap, index 1 have height and index 2 weight.

### anon-pcd-opensource

This Exploratory Data Analysis (EDA) notebook provide the data analysis of the [anon-pcd-opensource](https://ml.azure.com/dataset/anon-pcd-opensource/latest/details?wsid=/subscriptions/9b82ecea-6780-4b85-8acf-d27d79028f07/resourceGroups/cgm-ml-prod/providers/Microsoft.MachineLearningServices/workspaces/cgm-azureml-prod&tid=006dabd7-456d-465b-a87f-f7d557e319c8). The dataset contains 4858 scans of children in pcd format.

### anon-depthmap-95k

This Exploratory Data Analysis (EDA) notebook provide the data analysis of the [anon-depthmap-95k](https://ml.azure.com/dataset/anon-depthmap-95k/latest/details?wsid=/subscriptions/9b82ecea-6780-4b85-8acf-d27d79028f07/resourceGroups/cgm-ml-prod/providers/Microsoft.MachineLearningServices/workspaces/cgm-azureml-prod&tid=006dabd7-456d-465b-a87f-f7d557e319c8). The dataset contains 4858 scans of children in depthmap format. The files are in pickle format and the 0 index contains the depthmap, index 1 have height and index 2 weight.

### anon-realtime-evaluation

This Exploratory Data Analysis (EDA) notebook provide the data analysis of the [anon-realtime-evaluation](https://ml.azure.com/dataset/anon-realtime-evaluation/latest/details?wsid=/subscriptions/9b82ecea-6780-4b85-8acf-d27d79028f07/resourceGroups/cgm-ml-prod/providers/Microsoft.MachineLearningServices/workspaces/cgm-azureml-prod&tid=006dabd7-456d-465b-a87f-f7d557e319c8). The dataset contains 3000 scans of children in depthmap, Pointclouds and RGB format.

### anon-rgbd-5k

This Exploratory Data Analysis (EDA) notebook provide the data analysis of the [anon-rgbd-5k](https://ml.azure.com/dataset/anon-rgbd-5k/latest/details?wsid=/subscriptions/9b5bbfae-d5d1-4aae-a2ca-75159c0c887d/resourceGroups/cgm-ml-prod-we-rg/providers/Microsoft.MachineLearningServices/workspaces/cgm-ml-prod-we-azml&tid=3a27c573-ec1a-4734-9cd3-3208af51794b). The dataset contains 5534 scans of children in RGBD format.


## Getting started

### Requirements

Please refer to the home page of this repo to download the `requirements.txt` to set up you environment for running this notebook.

### Dataset and labels access

Data access is provided on as-needed basis following signature of the Welthungerhilfe Data Privacy & Commitment to
Maintain Data Secrecy Agreement. If you need data access (e.g. to train your machine learning models),
please contact [Markus Matiaschek](mailto:info@childgrowthmonitor.org) for details.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Versioning

Our [releases](https://github.com/Welthungerhilfe/cgm-ml/releases) use [semantic versioning](http://semver.org). You can find a chronologically ordered list of notable changes in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details and refer to [NOTICE](NOTICE) for additional licensing notes and use of third-party components.
