import numpy as np

from libs.ops import *


class DCGANGenerator(object):

  def __init__(self, hidden_dim=128, batch_size=64, hidden_activation=tf.nn.relu, output_activation=tf.nn.tanh, use_batch_norm=True, scope='generator', **kwargs):
    self.hidden_dim = hidden_dim
    self.batch_size = batch_size
    self.hidden_activation = hidden_activation
    self.output_activation = output_activation
    self.use_batch_norm = use_batch_norm
    self.scope = scope

  def __call__(self, z, is_training=True, **kwargs):
    with tf.variable_scope(self.scope):
      if self.use_batch_norm:
        l0  = self.hidden_activation(batch_norm(linear(z, 4 * 4 * 512, name='l0', stddev=0.02), name='bn0', is_training=is_training))
        l0  = tf.reshape(l0, [self.batch_size, 4, 4, 512])
        dc1 = self.hidden_activation(batch_norm(deconv2d( l0, [self.batch_size,  8,  8, 256], name='dc1', stddev=0.02), name='bn1', is_training=is_training))
        dc2 = self.hidden_activation(batch_norm(deconv2d(dc1, [self.batch_size, 16, 16, 128], name='dc2', stddev=0.02), name='bn2', is_training=is_training))
        dc3 = self.hidden_activation(batch_norm(deconv2d(dc2, [self.batch_size, 32, 32,  64], name='dc3', stddev=0.02), name='bn3', is_training=is_training))
        dc4 = self.output_activation(deconv2d(dc3, [self.batch_size, 32, 32, 3], 3, 3, 1, 1, name='dc4', stddev=0.02))
      else:
        l0  = self.hidden_activation(linear(z, 4 * 4 * 512, name='l0', stddev=0.02))
        l0  = tf.reshape(l0, [self.batch_size, 4, 4, 512])
        dc1 = self.hidden_activation(deconv2d(l0, [self.batch_size, 8, 8, 256], name='dc1', stddev=0.02))
        dc2 = self.hidden_activation(deconv2d(dc1, [self.batch_size, 16, 16, 128], name='dc2', stddev=0.02))
        dc3 = self.hidden_activation(deconv2d(dc2, [self.batch_size, 32, 32, 64], name='dc3', stddev=0.02))
        dc4 = self.output_activation(deconv2d(dc3, [self.batch_size, 32, 32, 3], 3, 3, 1, 1, name='dc4', stddev=0.02))
      x = dc4
    return x

  def generate_noise(self):
    return np.random.randn(self.batch_size, self.hidden_dim)


class SNDCGAN_Discrminator(object):

  def __init__(self, batch_size=64, hidden_activation=lrelu, output_dim=1, scope='critic', **kwargs):
    self.batch_size = batch_size
    self.hidden_activation = hidden_activation
    self.output_dim = output_dim
    self.scope = scope

  def __call__(self, x, update_collection=tf.GraphKeys.UPDATE_OPS, **kwargs):
    with tf.variable_scope(self.scope):
      c0_0 = self.hidden_activation(conv2d(   x,  64, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_0'))
      c0_1 = self.hidden_activation(conv2d(c0_0, 128, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_1'))
      c1_0 = self.hidden_activation(conv2d(c0_1, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_0'))
      c1_1 = self.hidden_activation(conv2d(c1_0, 256, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_1'))
      c2_0 = self.hidden_activation(conv2d(c1_1, 256, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_0'))
      c2_1 = self.hidden_activation(conv2d(c2_0, 512, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_1'))
      c3_0 = self.hidden_activation(conv2d(c2_1, 512, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c3_0'))
      c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
      l4 = linear(c3_0, self.output_dim, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='l4')
    return tf.reshape(l4, [-1])
