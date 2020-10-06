import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import math
import matplotlib.pyplot as plt
import os


class GradCAM:
    def __init__(self, model, layerName):
        self.model = model
        self.layerName = layerName

        self.gradModel = Model(inputs=[self.model.inputs],
                               outputs=[self.model.get_layer(self.layerName).output, model.output])

    def compute_heatmap(self, image, classIdx, eps=1e-8):

        with tf.GradientTape() as tape:
            tape.watch(self.gradModel.get_layer(self.layerName).output)
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = self.gradModel(inputs)
            if len(predictions) == 1:
                loss = predictions[0]
            else:
                loss = predictions[:, classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("float32")
        return heatmap


def make_grid(image_dir):
    from glob import glob
    files = glob(image_dir + '/*.png')
    result_figsize_resolution = 80
    images_count = 8
    # Calculate the grid size:
    grid_size = math.ceil(math.sqrt(images_count))

    # Create plt plot:
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(result_figsize_resolution, result_figsize_resolution))
    current_file_number = 0
    samples = files[:images_count]
    for image in samples:
        x_position = current_file_number % grid_size
        y_position = current_file_number // grid_size
        plt_image = plt.imread(image)
        axes[x_position, y_position].imshow(plt_image)
    # print((current_file_number + 1), '/', images_count, ': ', image_filename)

        current_file_number += 1

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    save_location = '{}/grid'.format(image_dir)
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    plt.savefig('{}/resultgrid.png'.format(save_location))
    plt.clf()
    for file in files:
        os.remove(file)
