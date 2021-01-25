#!/usr/bin/env bash

set -euox pipefail

python eval_main.py --qa_config_module qa_config_height_no_dropout
python eval_main.py --qa_config_module qa_config_height_dropout
# python eval_main.py --qa_config_module qa_config_filter
# python eval_main.py --qa_config_module qa_config_height
# python eval_main.py --qa_config_module qa_config_weight
