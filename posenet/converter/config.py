import yaml
import os



def load_config(config_name='config.yaml'):
    cfg_f = open(os.path.join('/home/prajwal/cgm-ml/posenet/converter', config_name), "r+")
    cfg = yaml.load(cfg_f)
    return cfg
