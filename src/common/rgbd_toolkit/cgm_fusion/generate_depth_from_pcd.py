import sys
sys.path.insert(0, "..")

#from cgm_fusion import calibration
from cgm_fusion import fusion
from cgm_fusion import utility

import numpy



# import glob, os
# os.chdir("/localssd/qrcode/")
# for file in glob.glob("*.ply"):
#     print(file)


import sys
sys.path.insert(0, "..")
# import dbutils
from cgmcore import utils
import matplotlib.pyplot as plt
import pandas as pd
import glob2
import os
# import config
from tqdm import tqdm
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import display, clear_output
import itertools
from tqdm import tqdm
import numpy as np


from matplotlib import pyplot as plt


# # get the number of rgb artifacts
# select_sql_statement = "SELECT path FROM artifact WHERE type='pcd';"
# pcd_paths = db_connector.execute(select_sql_statement, fetch_all=True)[0][0]
# print(pcd_paths)


# fusion.get_depth_image_from_point_cloud(calibration_file="dummy", pcd_file="/tmp/cloud_debug.ply", output_file="dummy")
# utility.get_depth_channel(ply_path="/tmp/cloud_debug.ply", output_path_np = "/tmp/output.npy", output_path_png="/tmp/output.png")
# utility.get_rgbd_channel(ply_path="/tmp/cloud_debug.ply", output_path_np = "/tmp/output.npy")

utility.get_all_channel(ply_path="/tmp/cloud_debug.ply", output_path_np = "/tmp/output.npy")


# utility.get_viz_channel(ply_path="/tmp/cloud_debug.ply",  channel=4, output_path="/tmp/red.png")