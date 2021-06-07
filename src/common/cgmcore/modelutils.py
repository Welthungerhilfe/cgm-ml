#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Tristan Behrens <tristan@ai-guru.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


"""
Helper module for Neural Networks.
"""

from tensorflow.keras import models, layers
import numpy as np
import tensorflow as tf
import os
import logging
import logging.config
import pickle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


def create_multiview_model(base_model, multiviews_num, input_shape, output_size, use_lstm):

    assert base_model == "voxnet" or base_model == "pointnet"

    if base_model == "voxnet":
        base_model = create_voxnet_model_homepage(input_shape, output_size)
    elif base_model == "pointnet":
        base_model = create_point_net(input_shape, output_size)

    input = layers.Input(shape=(multiviews_num,) + input_shape)

    multiview_outputs = []
    for i in range(multiviews_num):
        multiview_input = layers.Lambda(lambda x: x[:, i])(input)
        multiview_output = base_model(multiview_input)
        multiview_outputs.append(multiview_output)
    output = layers.Average()(multiview_outputs)

    model = models.Model(input, output)
    return model


def create_multiview_model_old(base_model, multiviews_num, input_shape, output_size, use_lstm):

    assert base_model == "voxnet" or base_model == "pointnet"

    if base_model == "voxnet":
        base_model = create_voxnet_model_homepage(input_shape, output_size)
    elif base_model == "pointnet":
        base_model = create_point_net(input_shape, output_size)

    model = models.Sequential()
    model.add(layers.TimeDistributed(base_model, input_shape=(multiviews_num,) + input_shape))
    if use_lstm is True:
        model.add(layers.LSTM(8, activation="relu"))
    else:
        model.add(layers.AveragePooling1D(multiviews_num))
        model.add(layers.Flatten())
    model.add(layers.Dense(output_size))

    return model


def create_dense_model(input_shape, output_size):
    """
    Creates a simple dense model.

    Note: This is only suitable for baseling.

    Args:
        input_shape (shape): Input-shape.
        output_size (int): Output-size.

    Returns:
        Model: A model.
    """

    model = models.Sequential(name="baseline-dense")
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(output_size))

    return model


def create_voxnet_model_small(input_shape, output_size):
    """
    Creates a small VoxNet.

    See: http://dimatura.net/publications/3dcnn_lz_maturana_scherer_icra15.pdf

    Args:
        input_shape (shape): Input-shape.
        output_size (int): Output-size.

    Returns:
        Model: A model.
    """

    #Trainable params: 301,378
    model = models.Sequential(name="C7-F32-P2-C5-F64-P2-D512")
    model.add(layers.Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(layers.Conv3D(32, (7, 7, 7), activation="relu"))
    model.add(layers.MaxPooling3D((4, 4, 4)))
    model.add(layers.Conv3D(64, (5, 5, 5), activation="relu"))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(output_size))

    return model


def create_voxnet_model_big(input_shape, output_size):
    """
    Creates a big VoxNet.

    See: http://dimatura.net/publications/3dcnn_lz_maturana_scherer_icra15.pdf

    Args:
        input_shape (shape): Input-shape.
        output_size (int): Output-size.

    Returns:
        Model: A model.
    """

    # Trainable params: 7,101,442
    model = models.Sequential(name="C7-F64-P4-D512")
    model.add(layers.Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(layers.Conv3D(64, (7, 7, 7), activation="relu"))
    model.add(layers.MaxPooling3D((4, 4, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(output_size))

    return model


def create_voxnet_model_homepage(input_shape, output_size):
    """
    Creates a small VoxNet.

    See: http://dimatura.net/publications/3dcnn_lz_maturana_scherer_icra15.pdf

    Note: This is the latest model that the VoxNet-authors used.

    Args:
        input_shape (shape): Input-shape.
        output_size (int): Output-size.

    Returns:
        Model: A model.
    """

    # Trainable params: 916,834
    model = models.Sequential(name="VoxNetHomepage")
    model.add(layers.Reshape(target_shape=input_shape + (1,), input_shape=input_shape))
    model.add(layers.Conv3D(32, (5, 5, 5), strides=(2, 2, 2), activation="relu"))
    model.add(layers.Conv3D(32, (3, 3, 3), strides=(1, 1, 1), activation="relu"))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(output_size))

    return model


def create_point_net(input_shape, output_size, hidden_sizes=[512, 256], use_lambda=False):
    """
    Creates a PointNet.

    See https://github.com/garyloveavocado/pointnet-keras/blob/master/train_cls.py

    Args:
        input_shape (shape): Input-shape.
        output_size (int): Output-size.

    Returns:
        Model: A model.
    """

    logging.info('Input Shape: %s', str(input_shape))

    num_points = input_shape[0]

    def mat_mul(A, B):
        result = tf.matmul(A, B)
        return result

    input_points = layers.Input(shape=input_shape)
    x = layers.Convolution1D(64, 1, activation='relu', input_shape=input_shape)(input_points)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution1D(1024, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=num_points)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(9, weights=[np.zeros([256, 9]), np.array(
        [1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = layers.Reshape((input_shape[1], input_shape[1]))(x)

    # forward net
    if use_lambda is True:
        g = layers.Lambda(mat_mul, arguments={'B': input_T})(input_points)
    else:
        g = layers.dot([input_points, input_T], axes=-1, normalize=True)
    g = layers.Convolution1D(64, 1, input_shape=input_shape, activation='relu')(input_points)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(64, 1, input_shape=input_shape, activation='relu')(g)
    g = layers.BatchNormalization()(g)

    # feature transform net
    f = layers.Convolution1D(64, 1, activation='relu')(g)
    f = layers.BatchNormalization()(f)
    f = layers.Convolution1D(128, 1, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Convolution1D(1024, 1, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPooling1D(pool_size=num_points)(f)
    f = layers.Dense(512, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dense(256, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dense(64 * 64, weights=[np.zeros([256, 64 * 64]),
                                       np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = layers.Reshape((64, 64))(f)

    # forward net
    if use_lambda is True:
        g = layers.Lambda(mat_mul, arguments={'B': feature_T})(g)
    else:
        g = layers.dot([g, feature_T], axes=-1, normalize=True)
    g = layers.Convolution1D(64, 1, activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(128, 1, activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(1024, 1, activation='relu')(g)
    g = layers.BatchNormalization()(g)

    # global_feature
    global_feature = layers.MaxPooling1D(pool_size=num_points)(g)

    # point_net_cls
    c = global_feature
    for hidden_size in hidden_sizes:
        c = layers.Dense(hidden_size, activation='relu')(c)
        c = layers.BatchNormalization()(c)
        c = layers.Dropout(rate=0.3)(c)

    c = layers.Dense(output_size, activation='linear')(c)
    prediction = layers.Flatten()(c)

    model = models.Model(inputs=input_points, outputs=prediction)
    return model


def create_dense_net(input_shape, output_size, hidden_sizes=[]):

    model = models.Sequential()

    # Input layer.
    model.add(layers.Flatten(input_shape=input_shape))

    for hidden_size in hidden_sizes:
        model.add(layers.Dense(hidden_size, activation="relu"))

    # Output layer.
    model.add(layers.Dense(output_size))

    return model


def create_2dCNN(input_shape, output_size):
    """
    Creates a 2dCNN.

    Args:
        input_shape (shape): Input-shape.
        output_size (int): Output-size.

    Returns:
        Model: A model.
    """

    # Trainable params: 13,086,913

    model_cnn = models.Sequential(name="2dCNN")
    model_cnn.add(layers.Conv2D(
        32, (3, 3),
        activation="relu",
        input_shape=(input_shape)))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Flatten())
    model_cnn.add(layers.Dense(512, activation="relu"))
    model_cnn.add(layers.Dropout(0.25))
    model_cnn.add(layers.Dense(output_size, activation="relu"))

    return model_cnn


def create_vgg(input_shape, output_size):
    """
   Creates a vgg19 and add Dense Layers.

   Args:
       input_shape (shape): Input-shape.
       output_size (int): Output-size.

   Returns:
       Model: A model.
   """

    # Trainable params: 68,229,825
    # Non-trainable params: 20,024,384
#     vgg19 = applications.VGG19(include_top=False, input_shape=(256, 256, 3))
#     vgg19.trainable = False
#     model_vgg = models.Sequential()
#     model_vgg.add(vgg19)
#     model_vgg.add(layers.Flatten())
#     model_vgg.add(layers.Dense(2048, activation='relu'))
#     model_vgg.add(layers.Dense(512, activation='relu'))
#     model_vgg.add(layers.Dense(128, activation='relu'))
#     model_vgg.add(layers.Dense(32, activation='relu'))
#     model_vgg.add(layers.Dense(1, activation='relu')) #regression

#     return model_vgg


# Method for saving model and history.
def save_model_and_history(output_path, datetime_string, model, history, training_details, name):

    logging.info("Saving model and history...")

    # Try to save model. Could fail.
    try:
        model_name = datetime_string + "-" + name + "-model.h5"
        model_path = os.path.join(output_path, model_name)
        model.save(model_path)
        logging.info("Saved model to %s", model_path)
    except Exception:
        logging.info("WARNING! Failed to save model. Use model-weights instead.")
        pass

    # Save the model weights.
    model_weights_name = datetime_string + "-" + name + "-model-weights.h5"
    model_weights_path = os.path.join(output_path, model_weights_name)
    model.save_weights(model_weights_path)
    logging.info("Saved model weights to %s", model_weights_path)

    # Save the training details.
    training_details_name = datetime_string + "-" + name + "-details.p"
    training_details_path = os.path.join(output_path, training_details_name)
    pickle.dump(training_details, open(training_details_path, "wb"))
    logging.info("Saved training details to %s", training_details_path)

    # Save the history.
    history_name = datetime_string + "-" + name + "-history.p"
    history_path = os.path.join(output_path, history_name)
    pickle.dump(history.history, open(history_path, "wb"))
    logging.info("Saved history to %s", history_path)


def load_pointnet(weights_path, input_shape, output_size, hidden_sizes):
    try:
        model = create_point_net(input_shape, output_size, hidden_sizes, use_lambda=False)
        model.load_weights(weights_path)
    except Exception:
        model = create_point_net(input_shape, output_size, hidden_sizes, use_lambda=True)
        model.load_weights(weights_path)
        return model
