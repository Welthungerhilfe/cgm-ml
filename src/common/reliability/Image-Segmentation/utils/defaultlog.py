import logging
import os
import sys
from logging import Formatter, StreamHandler, getLogger

def getlog():
    """ configure root logger from sys.path defaultlog OR this file
    enables sub-processes to use default log settings that can be easily overridden
    :return: configured root logger
    """
    try:
        # avoid recursion loop if already on pythonpath
        if os.path.dirname(__file__) not in sys.path:
            from defaultlog import log

            return log
    except:
        pass
    fmt = Formatter(
        fmt="[{module}:{lineno}:{levelname}]:{message} (time={asctime} {processName})",
        datefmt="%b-%d %H:%M",
        style="{",
    )
    # root log
    log = getLogger()
    stream = StreamHandler()
    stream.setFormatter(fmt)
    log.handlers.clear()
    log.addHandler(stream)
    log.setLevel(logging.INFO)
    log.info(f"logging started from {__file__}")

    # warning
    for name in ["paramiko", "pika", "fs.googledrivefs", "google_auth_httplib2"]:
        getLogger(name).setLevel(logging.WARNING)

    return log

log = getlog()
