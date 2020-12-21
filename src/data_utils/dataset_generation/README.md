# Create dataset utility

Dataset creation utility is used to create datasets for model training and evaluation.

## Overview

CGM-ML infrastructre currently prepares the dataset into 3 formats(RGB,pointclouds,depthmaps). PCD file is a standardised format. Depthmap is our own format developed for high compressed data.

In the future we plan to not support PCD files anymore (due to their big size).

## Getting started

To get started with this utility please follow the installation instruction mention on the main page of the github repository.

## Usage

The utility  looks for the configuration parameters provided in the `parameters.yaml`.
- `db_connection_file`: connection file to connect to the database 
- `scan_group`: type of dataset you want to prepare ['train','test']
- `scan_amount` : Amount of the data that you want to process.
- `source_path` : Source path from where the utility will get the data to process
- `target_path`: Target paths to storte the prepared data.
- `calibration` : Location of the calibratin file of the device

After setting the required configuration,simply run 

```
python create_dataset.py
```

## Converting depthmaps into PCD data

* Teh utility is using depthmap_toolkit to convert pointcloud data to depthmaps and they are stored in a pickle file along with many other labels.

### TODO

- [ ] Functionality to check the already processed qrcodes with non processed qrcodes  
- [ ] Automate the data registering functionality in Azure environment
- [ ] Embedd this fucntionality in the pipeline.


## CAUTION:

This toolkit only prepare the datasets. Please make sure to update the database from you end if you are using the datsets created by this toolkit. This toolkit does not ensure any update to the backend database for any operation perform on the prepared dataset 
