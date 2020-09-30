import unittest
from cgmcore import modelutils
import os
import tensorflow as tf
import numpy as np

class TestModels(unittest.TestCase):

    @unittest.skip("demonstrating skipping")
    def test_networks(self):
        """
        Tests all models.
        """

        input_shapes = [
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (30000, 3)
        ]
        output_size = 2

        creation_methods = [
            modelutils.create_dense_model,
            modelutils.create_voxnet_model_small,
            modelutils.create_voxnet_model_big,
            modelutils.create_voxnet_model_homepage,
            modelutils.create_point_net
        ]

        for input_shape, create_model in zip(input_shapes, creation_methods):
            model = create_model(input_shape, output_size)
            model_weights_path = "test.h5"
            model.save_weights(model_weights_path)
            model = create_model(input_shape, output_size)
            model.load_weights(model_weights_path)


    @unittest.skip("demonstrating skipping")
    def test_sequence_networks_lstm(self):

        model = modelutils.create_multiview_model(
            base_model="voxnet",
            multiviews_num=8,
            input_shape=(32, 32, 32),
            output_size = 2,
            use_lstm=True
            )
        print(model.inputs[0].shape)
        model.summary()

        model = modelutils.create_multiview_model(
            base_model="pointnet",
            multiviews_num=8,
            input_shape=(30000, 3),
            output_size = 2,
            use_lstm=True
            )
        print(model.inputs[0].shape)
        model.summary()

    def test_sequence_networks_no_lstm(self):

        model = modelutils.create_multiview_model(
            base_model="voxnet",
            multiviews_num=8,
            input_shape=(32, 32, 32),
            output_size = 2,
            use_lstm=False
            )
        print(model.inputs[0].shape)
        model.summary()
        prediction = model.predict(np.random.random((1, 8, 32, 32, 32)))
        self.assertEqual(prediction.shape, (1, 2))

        model = modelutils.create_multiview_model(
            base_model="pointnet",
            multiviews_num=8,
            input_shape=(30000, 3),
            output_size = 2,
            use_lstm=False
            )
        print(model.inputs[0].shape)
        model.summary()
        prediction = model.predict(np.random.random((1, 8, 30000, 3)))
        self.assertEqual(prediction.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()
