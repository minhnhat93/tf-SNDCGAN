import numpy as np
import tensorflow as tf

from libs.sn import spectral_normed_weight


def scope_has_variables(scope):
  return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0


def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
           name="conv2d", spectral_normed=False, update_collection=None, with_w=False, padding="SAME"):
  # Glorot intialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_h * k_w * output_dim
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    if spectral_normed:
      conv = tf.nn.conv2d(input_, spectral_normed_weight(w, update_collection=update_collection),
                          strides=[1, d_h, d_w, 1], padding=padding)
    else:
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
      return conv, w, biases
    else:
      return conv


def deconv2d(input_, output_shape,
             k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
             name="deconv2d", spectral_normed=False, update_collection=None, with_w=False, padding="SAME"):
  # Glorot initialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_h * k_w * output_shape[-1]
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    if spectral_normed:
      deconv = tf.nn.conv2d_transpose(input_, spectral_normed_weight(w, update_collection=update_collection),
                                      output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1], padding=padding)
    else:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    if with_w:
      return deconv, w, biases
    else:
      return deconv


def lrelu(x, leak=0.1):
  return tf.maximum(x, leak * x)


def linear(input_, output_size, name="linear", spectral_normed=False, update_collection=None, stddev=None, bias_start=0.0, with_biases=True,
           with_w=False):
  shape = input_.get_shape().as_list()

  if stddev is None:
    stddev = np.sqrt(1. / (shape[1]))
  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    weight = tf.get_variable("w", [shape[1], output_size], tf.float32,
                             tf.truncated_normal_initializer(stddev=stddev))
    if with_biases:
      bias = tf.get_variable("b", [output_size],
                             initializer=tf.constant_initializer(bias_start))
    if spectral_normed:
      mul = tf.matmul(input_, spectral_normed_weight(weight, update_collection=update_collection))
    else:
      mul = tf.matmul(input_, weight)
    if with_w:
      if with_biases:
        return mul + bias, weight, bias
      else:
        return mul, weight, None
    else:
      if with_biases:
        return mul + bias
      else:
        return mul


def batch_norm(input, is_training=True, momentum=0.9, epsilon=2e-5, in_place_update=True, name="batch_norm"):
  if in_place_update:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        updates_collections=None,
                                        is_training=is_training,
                                        scope=name)
  else:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        is_training=is_training,
                                        scope=name)
