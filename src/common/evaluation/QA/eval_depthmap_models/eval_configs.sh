#!/usr/bin/env bash

set -euox pipefail

python eval_main.py --qa_config_module qa_config_mcnn
python eval_main.py --qa_config_module qa_config_height_no_dropout
python eval_main.py --qa_config_module qa_config_height_dropout
# python eval_main.py --qa_config_module qa_config_filter
# python eval_main.py --qa_config_module qa_config_height
python eval_main.py --qa_config_module qa_config_weight_no_dropout
python eval_main.py --qa_config_module qa_config_weight_dropout

# Combine results
python combine_results.py --model_measurement height

# Find inaccurate results for different models
python inaccurate_scans.py
