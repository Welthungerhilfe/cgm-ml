from tensorflow.keras import models, layers


def relu_bn(inputs):
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def residual_block(x, downsample, filters, kernel_size=3, dropout=0.0):
    y = layers.Conv2D(kernel_size=kernel_size,
                      strides=(1 if not downsample else 2),
                      filters=filters,
                      padding="same")(x)
    y = relu_bn(y)
    y = layers.Conv2D(kernel_size=kernel_size,
                      strides=1,
                      filters=filters,
                      padding="same")(y)

    if downsample:
        x = layers.Conv2D(kernel_size=1,
                          strides=2,
                          filters=filters,
                          padding="same")(x)
    out = layers.Add()([x, y])
    out = relu_bn(out)
    out = layers.Dropout(dropout)(out)
    return out


def create_res_net(input_shape, num_blocks_list, dropouts_list):

    inputs = layers.Input(shape=input_shape)
    num_filters = 64

    t = layers.BatchNormalization()(inputs)
    t = layers.Conv2D(kernel_size=3,
                      strides=1,
                      filters=num_filters,
                      padding="same")(t)
    t = relu_bn(t)

    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        dropout = dropouts_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters, dropout=dropout)
        num_filters *= 2

    t = layers.AveragePooling2D(4)(t)
    t = layers.Flatten()(t)
    outputs = layers.Dense(1, activation='linear')(t)

    model = models.Model(inputs, outputs)

    return model
