import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K
from .layers import KNN, Transform, MultiGraphAttention, GraphAttention
from .utils import assert_shape_is


class GAPNet(tf.keras.Model):

    def __init__(self, number_of_points=1024, features_in=3, k=20, features_out=16, **kwargs):
        super(GAPNet, self).__init__()

        self.number_of_points = number_of_points
        self.features_in = features_in
        self.k = k
        self.features_out = features_out

        self.build_graph(input_shape=(None, number_of_points, features_in))

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)

    def build(self, input_shape, **kwargs):

        # Create attention layer with one head.
        self.onehead_attention = MultiGraphAttention(k=self.k, features_out=self.features_out, heads=1)

        # Create spatial transform layer.
        self.transform = Transform()

        # Create attention layer with four heads.
        self.fourhead_attention = MultiGraphAttention(k=self.k, features_out=self.features_out, heads=4)

        # MLP 1 on attention features.
        self.mlp1 = layers.Dense(64, activation="linear")
        self.mlp_bn1 = layers.BatchNormalization()
        self.mlp_activation1 = layers.Activation("relu")

        # MLP 2 on attention features.
        self.mlp2 = layers.Dense(64, activation="linear")
        self.mlp_bn2 = layers.BatchNormalization()
        self.mlp_activation2 = layers.Activation("relu")

        # MLP 3 on attention features.
        self.mlp3 = layers.Dense(64, activation="linear")
        self.mlp_bn3 = layers.BatchNormalization()
        self.mlp_activation3 = layers.Activation("relu")

        # MLP 4 on attention features.
        self.mlp4 = layers.Dense(128, activation="linear")
        self.mlp_bn4 = layers.BatchNormalization()
        self.mlp_activation4 = layers.Activation("relu")

        # MLP 5.
        self.mlp5 = layers.Dense(1024, activation="linear")
        self.mlp_bn5 = layers.BatchNormalization()
        self.mlp_activation5 = layers.Activation("relu")

        # Flatten.
        self.flatten = layers.Flatten()

        # Dense 1.
        self.dense1 = layers.Dense(512, activation="relu")
        self.dense_dropout1 = layers.Dropout(0.5)

        # Dense 2.
        self.dense2 = layers.Dense(256, activation="relu")
        self.dense_dropout2 = layers.Dropout(0.5)

        # Dense 3.
        self.dense3 = layers.Dense(1, activation="linear")

        super(GAPNet, self).build(input_shape)

    def call(self, inputs):

        point_cloud = inputs
        self.point_cloud_in = point_cloud
        assert_shape_is(point_cloud, (1024, 3))

        # First attention layer with one head.
        onehead_attention = self.onehead_attention(point_cloud)
        onehead_attention_features = onehead_attention[0]
        onehead_graph_features = onehead_attention[1]
        onehead_attention_coefficients = onehead_attention[2]
        self.onehead_attention_coefficients_out = onehead_attention_coefficients
        assert_shape_is(onehead_attention_features, (1024, 1, 16))
        assert_shape_is(onehead_graph_features, (1024, 20, 16))
        assert_shape_is(onehead_attention_coefficients, (1024, 1, 20))

        # Skip connection from point cloud to attention features.
        point_cloud_expanded = K.expand_dims(point_cloud, axis=2)
        assert_shape_is(point_cloud_expanded, (1024, 1, 3))
        onehead_attention_features = K.concatenate([onehead_attention_features, point_cloud_expanded])
        assert_shape_is(onehead_attention_features, (1024, 1, 19))
        del point_cloud_expanded

        # Spatial transform.
        point_cloud_transformed = self.transform([point_cloud, onehead_attention_features, onehead_graph_features])
        assert_shape_is(point_cloud_transformed, (1024, 3))
        self.point_cloud_transformed_out = point_cloud_transformed
        del point_cloud

        # Second attention layer with four head.
        fourhead_attention = self.fourhead_attention(point_cloud_transformed)
        fourhead_attention_features = fourhead_attention[0]
        fourhead_graph_features = fourhead_attention[1]
        fourhead_attention_coefficients = fourhead_attention[2]
        self.fourhead_attention_coefficients_out = fourhead_attention_coefficients
        assert_shape_is(fourhead_attention_features, (1024, 1, 64))
        assert_shape_is(fourhead_graph_features, (1024, 20, 64))
        assert_shape_is(fourhead_attention_coefficients, (1024, 1, 80))

        # Skip connection from transformed point cloud to attention features.
        point_cloud_expanded = K.expand_dims(point_cloud_transformed, axis=2)
        assert_shape_is(point_cloud_expanded, (1024, 1, 3))
        onehead_attention_features = K.concatenate([fourhead_attention_features, point_cloud_expanded])
        assert_shape_is(onehead_attention_features, (1024, 1, 67))

        # MLP 1 on attention features.
        net1 = self.mlp1(onehead_attention_features)
        net1 = self.mlp_bn1(net1)
        net1 = self.mlp_activation1(net1)
        assert_shape_is(net1, (1024, 1, 64))

        # MLP 2 on attention features.
        net2 = self.mlp2(net1)
        net2 = self.mlp_bn2(net2)
        net2 = self.mlp_activation2(net2)
        assert_shape_is(net2, (1024, 1, 64))

        # MLP 3 on attention features.
        net3 = self.mlp3(net2)
        net3 = self.mlp_bn3(net3)
        net3 = self.mlp_activation3(net3)
        assert_shape_is(net3, (1024, 1, 64))

        # MLP 4 on attention features.
        net4 = self.mlp4(net3)
        net4 = self.mlp_bn4(net4)
        net4 = self.mlp_activation4(net4)
        assert_shape_is(net4, (1024, 1, 128))

        # Maximum for graph features.
        fourhead_graph_features_max = tf.reduce_max(fourhead_graph_features, axis=2, keepdims=True)
        assert_shape_is(fourhead_graph_features_max, (1024, 1, 64))

        # Concatenate all MLPs and maximum of graph features.
        net = layers.concatenate([net1, net2, net3, net4, fourhead_graph_features_max])
        assert_shape_is(net, (1024, 1, 384))

        # MLP 5.
        net = self.mlp5(net)
        net = self.mlp_bn5(net)
        net = self.mlp_activation5(net)
        assert_shape_is(net, (1024, 1, 1024))

        # Maximum for net.
        net = K.max(net, axis=1, keepdims=True)
        assert_shape_is(net, (1, 1, 1024))

        # Flatten.
        net = self.flatten(net)
        assert_shape_is(net, (1024,))

        # Dense 1.
        net = self.dense1(net)
        net = self.dense_dropout1(net)
        assert_shape_is(net, (512,))

        # Dense 2.
        net = self.dense2(net)
        net = self.dense_dropout2(net)
        assert_shape_is(net, (256,))

        # Dense 3.
        net = self.dense3(net)
        #assert_shape_is(net, (40,))

        return net

    def create_explaining_model(self):
        """
        Creates a neural network that has the auxilary outputs.
        """

        input = self.point_cloud_in
        outputs = [
            self.point_cloud_transformed_out,
            self.onehead_attention_coefficients_out,
            self.fourhead_attention_coefficients_out
        ]
        return models.Model(input, outputs)
