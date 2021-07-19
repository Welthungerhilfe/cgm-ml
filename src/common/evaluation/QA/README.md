# Evaluate model performance

## Quality Assurance

Inside `QA/`, we have implemented logic to evaluate different models and to perform evaluation of different use cases.

- The steps to evaluate the model are run in an azure pipeline (see `test-pipeline.yml`)
- This runs `src/common/evaluation/QA/eval_depthmap_models/eval_configs.sh` which contains the list of configs to run
- The azure pipeline will sporn AzureML runs for each config (for heavy computations)
- Once the job on the cluster is done, this gets the resulting model evaluation results (e.g., CSV, png, or similar)

## Evaluation on Depthmap Model: `eval_depthmap_models/`

It contains logic to perform model evaluations trained on single artifacts architecture.

## Steps to add a model evaluation

Each model evaluation contains a configuration
(e.g. [qa_config_height.py](./eval_depthmap_models/src/qa_config_height.py)).
It contains the following parameters:

    1. `MODEL_CONFIG` : Model specific parameters
        e.g. specify model to use for evaluation
    2. `EVAL_CONFIG` : Evaluation specific parameters
        e.g. name of the experiment and cluster name in which evaluation need to performed
    3. `DATA_CONFIG` : Dataset specific parameters
        e.g. dataset name registered in datastore for evaluation

If you want to add a model evaluation, do the following:
- provide a new config file `my_new_config.py`
- add a new step in `src/common/evaluation/QA/eval_depthmap_models/eval_configs.sh`: Add a line `python eval_main.py --qa_config_module my_new_config`

## Run a model evaluation

You can run the evaluation by triggering the pipeline [test-pipeline.yml](./test-pipeline.yml).

Make necessary changes and commit the code to run the evaluation.

## Run locally (without pipeline)

It is useful to set `DATA_CONFIG.NAME='anon-depthmap-mini'` for debugging purposes.

For debugging purposes, you might want to run this without the pipeline.
To do so, execute `eval_main.py`.
This will still use a node in the GPU cluster to do the heavy processing.

If you want to run everything locally, you can run `evaluate.py` locally, so the heavy processing is done locally.
