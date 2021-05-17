# Report for Experiment: 2021q1-depthmap-plaincnn-height-ensemble

This report summarizes our experiment, which uses depthmaps as input data
for height prediction. We use a Convolutional Neural Network (CNN).

The main advantage is that this model that **can measure its own uncertainty**, in order to make a **reliable** prediction.

## Resources

* https://markus-hinsche.medium.com/can-we-trust-in-ai-for-the-most-important-of-medical-decisions-1f2bda3ce039
* https://slideslive.com/38935801/practical-uncertainty-estimation-outofdistribution-robustness-in-deep-learning
* https://techxplore.com/news/2020-11-neural-network.html

## Approach

* Train (from scratch) 16 different models by starting with different random seeds, which are used for weight initialization and train/valid split
* During evaluation, load 16 different models and predict
* Calculate mean of all 16 predictions: This will be the height prediction
* Calculated standard deviation of all 16 predictions: This will express the model uncertainty

## Results

This approach achieved a `min(val_mae)` of approx `2cm` (
see [q1-ensemble-warmup](https://ml.azure.com/experiments/id/114b0f9d-4c3d-4a9d-b251-f38f321fdd2a?wsid=/subscriptions/9b5bbfae-d5d1-4aae-a2ca-75159c0c887d/resourceGroups/cgm-ml-prod-we-rg/providers/Microsoft.MachineLearningServices/workspaces/cgm-ml-prod-we-azml&tid=3a27c573-ec1a-4734-9cd3-3208af51794b)
)
- This was trained on all codes.

## Future work

* train again but exclude blacklist
* train again but choose different hyperparameters
