from etl.etl import ETL
import sys
import logging
import yaml
from logging.config import dictConfig

dictConfig(yaml.safe_load(open('etl/logging.yaml')))
log = logging.getLogger()

if __name__ == '__main__':
    config_path = sys.argv[1]
    log.info("The config path is %s" % config_path)
    etl_process = ETL(simulate="simulate" in sys.argv)
    etl_process.initialize(config_path)
    etl_process.run()
