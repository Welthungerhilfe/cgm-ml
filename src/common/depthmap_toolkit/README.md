# Depthmap toolkit

Depthmap toolkit is an utility to convert and visualise the data captured by cgm-scanner.

## Overview

CGM-Scanner currently captures the data as depthmaps and PCD files. PCD file is a standardised format. Depthmap is our own format developed for high compressed data.

In the future we plan to not support PCD files anymore (due to their big size).

## Tools

### Converting PCD data into depthmap

* The convertor uses `camera_calibration.txt` file which is the calibration from Lenovo Phab 2 Pro. Converting could be done by following command:

`python convertpcd2depth.py pcd_dir`

* The `pcd_dir` folder could contain more PCD files.
* The output will be stored in folder output.

### Converting depthmaps into PCD data

* The convertor accepts only the data captured by cgm-scanner. The data could be captured by any ARCore device supporting ToF sensor. Converting could be done by following command:

`python convertdepth2pcd.py depthmap_dir`

* The `depthmap_dir` folder has to contain camera_calibration.txt file and subfolder depth containing one or more depthmap files.
* The output will be stored in folder export.

### Visualisation of depthmaps

* The tool accepts only the data captured by cgm-scanner. The data could be captured by any ARCore device supporting ToF sensor. Tool could be opened by following command:

`python toolkit.py depthmap_dir`

* The `depthmap_dir` folder has to contain camera_calibration.txt file and subfolder depth containing one or more depthmap files.
* By arrows "<<" and ">>" you can switch to next or previous depthmap in the folder
* Export OBJ will export the data as a pointcloud into OBJ file in export folder, this data will be reoriented using depthmap pose (if available)
* Export PCDwill export the data as a pointcloud into PCD file in export folder
* Convert all PCDs button has the same functionality as convertdepth2pcd
