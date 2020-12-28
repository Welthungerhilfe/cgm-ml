
Pre-requisities:

Pretrained CAFFE Pose estimation model is executed on (1/6)th of the samples from the "anon_RGB_training dataset" and 
the result is stored in a json file. 
As the original result file is huge, a sample json file with the name "pose_estimation_output.json" is created and uploaded in the same directory as analyseOutput.py.

The objective of analyseOutput.py script is to analyse pose_estimation_output.json in order to find the missing posepoints and thereby calculating the accuracy.
 
Following files are required to run analyseOutput.py:

 1. config.py located under cgm-ml/src/common/reliability/pose_estimation/QA/src/config.py
 2. posepoints.py located under cgm-ml/src/common/reliability/pose_estimation/QA/src/posepoints.py
 3. pose_estimation_output.json (this is ths result of executing the notebook located in the directory, 
 cgm-ml/src/common/reliability/pose_estimation/QA/poseest_notebook.ipynb

- config.py file contains the list of parameter configuration that has been used to train the pose estimation model
- posepoints.py file contains functions to set pose details (e.g. POSE_PAIRS and BODY_PARTS) which has been used in the analyseOutput.py script

Required libraries to run analyseOutput.py:
- pandas

Tests have been carried on different samples of data and results have been verified. 
Please refer to SampleOutput.txt for a sample output.
