## Experiment config

Inside [config.py](./src/config.py), you can adjusts the parameters of the training.
This includes dataset configurations, e.g. `DATASET_MODE` chooses if you mount or download the dataset.
This also includes hyperparameters for the training.

## Run local training

```bash
cd Models/CNNDepthMap/CNNDepthMap-height/q3-depthmapmultiartifactlatefusion-plaincnn-height
python -m src.train
```

## Run unit tests

```bash
pytest
```
