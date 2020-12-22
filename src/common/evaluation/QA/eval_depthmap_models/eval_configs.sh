#!/usr/bin/env bash

set -euox pipefail

python eval_main.py --qa_config_module qa_config_42c4ef33
python eval_main.py --qa_config_module qa_config_cb44f6db
