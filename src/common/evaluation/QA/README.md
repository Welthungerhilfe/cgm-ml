# Evaluate model performance

## Quality Assurance

Inside QA, we have implemented logic to evaluate various different models and to perform evaluation of different use cases.

- The steps to evaluate the model are run in an azure pipeline
- A jupyter notebook is run (via papermill)
- the resulting notebook is a pipeline artifact
- the jupyter notebook spins up a node on an AzureML cluster evaluate
- once the job on the cluster is done, this gets the resulting model evaluation results (CSV) and displays it in a notebook cell

## Evaluation on Depthmap Model

It contains logic to perform model evaluations trained on single artifacts architecture.

## Evaluate the measure of Standardisation Test

It contains logic to evaluate acceptability of enumerators and our model based on measurement performed while standardisation Test.

## Steps to add a model evaluation

Each model evaluation contains a configuration (e.g. [qa_config_height.py](./eval_depthmap_models/src/qa_config_height.py)). It contains the following parameters:

    1. `MODEL_CONFIG` : Model specific parameters
        e.g. specify model to use for evaluation
    2. `EVAL_CONFIG` : Evaluation specific parameters
        e.g. name of the experiment and cluster name in which evaluation need to performed
    3. `DATA_CONFIG` : Dataset specific parameters
        e.g. dataset name registered in datastore for evaluation

If you want to add a model evaluation, do the following:
- provide a new config file `my_new_config.py`
- add a new step in [test-pipeline.yml](./test-pipeline.yml): Look for `python eval_main.py --qa_config_module my_new_config`

## Run a model evaluation

You can run the evaluation by triggering the pipeline [test-pipeline.yml](./test-pipeline.yml)

Make necessary changes and commit the code to run the evaluation.

## Run without pipeline

For debugging purposes, you might want to run this without the pipeline.
To do so, execute `eval_main.py`.
This will still use a node in the GPU cluster to do the heavy processing.

Also it is useful to set `DATA_CONFIG.NAME='anon-depthmap-mini'` for debugging purposes.
