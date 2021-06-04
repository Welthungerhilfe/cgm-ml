from cgm_fusion import utility
import sys
import logging
import logging.config
sys.path.insert(0, "..")

# flake8: noqa: E501

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


#from cgm_fusion import calibration


# import glob, os
# os.chdir("/localssd/qrcode/")
# for file in glob.glob("*.ply"):
#     logging.info(file)

sys.path.insert(0, "..")
# import dbutils
# import config


# # get the number of rgb artifacts
# select_sql_statement = "SELECT path FROM artifact WHERE type='pcd';"
# pcd_paths = db_connector.execute(select_sql_statement, fetch_all=True)[0][0]
# logging.info(pcd_paths)

# fusion.get_depth_image_from_point_cloud(calibration_file="dummy", pcd_file="/tmp/cloud_debug.ply", output_file="dummy")
# utility.get_depth_channel(ply_path="/tmp/cloud_debug.ply", output_path_np = "/tmp/output.npy", output_path_png="/tmp/output.png")
# utility.get_rgbd_channel(ply_path="/tmp/cloud_debug.ply", output_path_np = "/tmp/output.npy")

utility.get_all_channel(ply_path="/tmp/cloud_debug.ply",
                        output_path_np="/tmp/output.npy")

# utility.get_viz_channel(ply_path="/tmp/cloud_debug.ply",  channel=4, output_path="/tmp/red.png")
