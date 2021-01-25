# Data util

## Dataset Versioning concept

We store our datasets in AzureML.

AzureML datasets can have versions: When we create a new version of a dataset (e.g. because there was a bug in the dataset) we keep the existing versions exactly as they are, and create a new version.

The dataset can be reference in a model training and also in a model evaluation. Therefore these trainings and evaluations are reproducable because the dataset does not change.
