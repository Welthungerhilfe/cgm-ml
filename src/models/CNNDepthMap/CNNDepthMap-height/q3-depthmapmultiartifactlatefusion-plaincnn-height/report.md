# Report for Experiment: q3-depthmapmultiartifactlatefusion-plaincnn-height

This report summarizes our experiment, which uses depthmaps as input data
for height prediction. We use a Convolutional Neural Network (CNN).

## Related work

[Large-scale Video Classification with Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf) introduces the idea of early and late fusion.

![Fusion strategies](docs/fusion-strategies.jpg)
Figure: In the bottom, there is a video containing multiple frames. "Single Frame" only uses a single frame to make a prediction. "Late Fusion" uses multiple frames where each frames is processed by a network independently and only in the end (a.k.a. late) the results are combined to a single prediction.

## Approach

We use the "late fusion" idea in our approach.

### Sampling

We sample `N` artifacts from a scan.
We compare 2 different sampling strategies:

- sliding window
- systematic sampling

### Architecture

We divide the neural network into a **base network** and a **network head**.

The base network is shared by all the `N` artifacts.
Each artifact goes through the base network.
This can also be viewed as a feature extraction,
s.t. for each artifact features are extracted.
To combine the features of multiple artifacts, we concatenate all features.

The network head is composed of dense layers that should combine and weigh the features.

### Pretraining

Using the single artifact approach, we trained a network (Experiment is called `q3-depthmap-plaincnn-height`).
This network achieves a `min(val_mae)` of `1.96cm` to `2.21cm`.

We use all but the last layer(s) of this network to initialize our base network.

We initialize the network head randomly.

We freeze the base network in order to keep the well-trained parameters.

## Results

This baseline achieved a `min(val_mae)` of `1.96cm`.

This approach achieved a `min(val_mae)` of `0.83cm` (
see [q3-depthmapmultiartifactlatefusion-plaincnn-height-95k - Run 26](https://ml.azure.com/experiments/id/da5aef2b-b171-44bd-8480-749dcfdd5258/runs/q3-depthmapmultiartifactlatefusion-plaincnn-height-95k_1601382575_b8b06f8d?wsid=/subscriptions/9b82ecea-6780-4b85-8acf-d27d79028f07/resourceGroups/cgm-ml-prod/providers/Microsoft.MachineLearningServices/workspaces/cgm-azureml-prod&tid=006dabd7-456d-465b-a87f-f7d557e319c8)
)

## Future work

* the paper mentions that "Slow Fusion" gives the best results
