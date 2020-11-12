"""
setup jupyter for data analysis.
"""
import logging
# builtins
import os
import warnings
import pickle
from pathlib import Path

# analysis
# from IPython import get_ipython
from IPython.core.display import HTML
from IPython.display import display as d
from defaultlog import getlog, log

if log.getEffectiveLevel() > logging.DEBUG:
    warnings.filterwarnings("ignore")

HOME = Path(os.path.expanduser("~"))

def flog(text):
    """ for finding logging problems """
    with open("c:/flog.txt", "a") as f:
        f.write(str(text))

def resetlog():
    global log
    log = getlog()


# extensions
# try:
#     get_ipython().magic("load_ext autoreload")
# except:
#     print("")
# try:
#     get_ipython().magic("autoreload 2")  # autoreload all modules
# except:
#     print("")
# try:
#     get_ipython().magic("matplotlib inline")  # show charts inline
# except:
#     print("")
# try:
#     get_ipython().magic("load_ext cellevents")  # show time and alert
# except:
#     print("")


def reset():
    """ resets the kernel """
    HTML("<script>Jupyter.notebook.kernel.restart()</script>")

def wide():
    """ makes notebook fill screen width """
    d(HTML("<style>.container { width:100% !important; }</style>"))

def save(o, filename):
    """ pickle """
    filename = str(filename)
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(o, f)

def load(filename):
    """ unpickle """
    filename = str(filename)
    with open(filename + ".pkl", "rb") as f:
        return pickle.load(f)
