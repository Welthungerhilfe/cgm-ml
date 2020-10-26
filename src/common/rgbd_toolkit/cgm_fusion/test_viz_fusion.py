import sys
sys.path.insert(0, "..")

from cgm_fusion import calibration
from cgm_fusion import fusion
from cgm_fusion import utility

import numpy

utility.get_viz_depth("/tmp/cloud_debug.ply")
utility.get_viz_rgb("/tmp/cloud_debug.ply")
utility.get_viz_segmentation("/tmp/cloud_debug.ply")
utility.get_viz_normal_z('/tmp/cloud_debug.ply')
utility.get_viz_confidence("/tmp/cloud_debug.ply")
