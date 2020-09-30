from __future__ import absolute_import

# For importing stuff. Relative imports.
import sys
sys.path.append("..")
sys.path.append("../cgmcore")

import utils
from keras.models import load_model
import numpy as np
import pyntcloud
import pandas as pd
from flask import Flask, request, Response
import jsonpickle
import tensorflow as tf
from keras import backend as K


class CGMRegressor(object):
    """
    Regressor for the child-growth monitor.

    Predicts on pointclouds.
    """


    def __init__(self):
        model_path = utils.get_latest_model("../models", "voxnet")
        print(model_path)

        self.model = load_model(model_path)
        self.model.summary()
        self.graph = tf.get_default_graph()

        self.voxel_size_meters = 0.1


    def predict(self, point_clouds, target_names):
        """
        Predicts on a list of pointclouds.

        Each pointcloud is turned into a voxelgrid.
        The neural network then predicts the features.
        The features are then averaged.

        Args:
            point_clouds (ndarray): a array of pointcloud.
            target_names (list of strings): the name of the targets.

        Returns:
            type: description

        Raises:
            Exception: description
        """


        # Turn pointclouds into a voxelgrids.
        voxelgrids = []
        for point_cloud in point_clouds:
            dataframe = pd.DataFrame(point_cloud, columns=["x", "y", "z", "c"])
            point_cloud = pyntcloud.PyntCloud(dataframe)
            voxelgrid_id = point_cloud.add_structure("voxelgrid", size_x=self.voxel_size_meters, size_y=self.voxel_size_meters, size_z=self.voxel_size_meters)
            voxelgrid = point_cloud.structures[voxelgrid_id].get_feature_vector(mode="density")
            voxelgrid = utils.ensure_voxelgrid_shape(voxelgrid, (32, 32, 32))
            voxelgrids.append(voxelgrid)
        voxelgrids = np.array(voxelgrids)
        print(voxelgrids.shape)

        # Predict.
        with self.graph.as_default():

            # Predict and average.
            prediction =  self.model.predict(voxelgrids)
            prediction = np.mean(prediction, axis=0)

            # Build the response as dictionary.
            prediction = {
                "message": "success",
                target_names[0]: str(prediction[0]),
                target_names[1]: str(prediction[1])
            }

            # Done.
            return prediction


# Using flask to expose the service as a plain REST-server.
# Note: Just for testing. Hopefully seldon will do something similar.
if __name__ == "__main__":

    # Initialize the regressor.
    regressor = CGMRegressor()

    # Create a flask-app.
    app = Flask(__name__)

    # Allow for POSTing pointclouds.
    @app.route("/cgm/regressor", methods=["POST"])
    def flask_post():

        # Get the data from request and reshape.
        point_clouds = np.fromstring(request.data, np.float32)
        point_clouds = np.reshape(point_clouds, (-1, 30000, 4))
        print("Received:", point_clouds.shape)

        # Predict.
        prediction = regressor.predict(point_clouds, ["height", "weight"])

        # Go full JSON.
        response = jsonpickle.encode(prediction)

        # Respond.
        return Response(response=response, status=200, mimetype="application/json")

    # Finally, expose the service.
    app.run(host="0.0.0.0", port=5000)
