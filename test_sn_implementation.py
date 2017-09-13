import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from libs.sn import spectral_normed_weight
import timeit

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(np.random.normal(size=[784, 10], scale=0.02), name='W', dtype=tf.float32)
b = tf.Variable(tf.zeros([10]), name='b', dtype=tf.float32)
W_bar, sigma = spectral_normed_weight(W, num_iters=1, with_sigma=True, update_collection=SPECTRAL_NORM_UPDATE_OPS)

y = tf.nn.softmax(tf.matmul(x, W_bar) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

s, _, _ = tf.svd(W)
s_bar, _, _ = tf.svd(W_bar)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS)
for _ in range(1000):
  start = timeit.default_timer()
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  sigma_, s_, s_bar_ = sess.run([sigma, s, s_bar])
  # TESTING:
  # Check for s_[0] (largest singular value) - sigma
  # They are very close. Difference mostly around less than 5%
  # Also, svd of W_bar is close to 1
  # So I assume my implementation of singular value power iteration approximation is correct
  for update_op in update_ops:
    sess.run(update_op)
  stop = timeit.default_timer()
  print('Iteration:', _, 'Sigma: ', sigma_, 'Max SVD: ', s_[0], 'Max SVD approx: ', s_bar_[0], 'Percentage Difference: ', abs(s_[0] - sigma_) / s_[0] * 100, 'Time: ', stop - start)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
