import tensorflow as tf


def assert_shape_is(tensor, expected_shape):
    assert tensor.dtype == tf.float32, str(tensor.dtype)
    assert isinstance(tensor, tf.Tensor), type(tensor)
    assert isinstance(expected_shape, list) or isinstance(expected_shape, tuple), type(expected_shape)
    tensor_shape = tensor.shape[1:]
    if tensor_shape != expected_shape:
        raise Exception("{} is not equal {}".format(tensor_shape, expected_shape))
