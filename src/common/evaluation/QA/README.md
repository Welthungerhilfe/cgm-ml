## Quality Assurance

Inside QA, we have implemented logic to evaluate various different models and to perform evaluation of different use cases.

## Evaluation on Depthmap Height Model 

It contains logic to perform evaluation of models trained on single artifacts architecture.

## Evaluate the measure of Standardisation Test

It contains logic to evaluate acceptability of enumerators and our model based on measurement performed while standardisation Test.

## Steps to perform evaluation

Each evaluation contains the [test_config.py](./eval-depthmap-height/src/config.py) in src directory.

test_config.py mainly contains below parameters:

    1. `MODEL_CONFIG` : Model specific parameters
        e.g. specify model to use for evaluation
    2. `EVAL_CONFIG` : Evaluation specific parameters
        e.g. name of the experiment and cluster name in which evaluation need to performed
    3. `DATA_CONFIG` : Dataset specific parameters
        e.g. dataset name registered in datastore for evaluation

Make necessary changes and commit the code to run the evaluation.

For more details one can look the [test_config.py](./eval-depthmap-height/src/config.py)

