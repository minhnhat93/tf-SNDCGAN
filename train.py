import timeit

import numpy as np
import tensorflow as tf

from libs.input_helper import Cifar10
from libs.utils import save_images, mkdir
from net import DCGANGenerator, SNDCGAN_Discrminator
import _pickle as pickle
from libs.inception_score.model import get_inception_score

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_iter', 100000, '')
flags.DEFINE_integer('snapshot_interval', 1000, 'interval of snapshot')
flags.DEFINE_integer('evaluation_interval', 10000, 'interval of evalution')
flags.DEFINE_integer('display_interval', 100, 'interval of displaying log to console')
flags.DEFINE_float('adam_alpha', 0.0001, 'learning rate')
flags.DEFINE_float('adam_beta1', 0.5, 'beta1 in Adam')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam')
flags.DEFINE_integer('n_dis', 1, 'n discrminator train')

mkdir('tmp')

SPECTRAL_NORM_UPDATE_OPS = 'spectral_norm_update_ops'
INCEPTION_FILENAME = 'inception_score.pkl'
config = FLAGS.__flags
generator = DCGANGenerator(**config)
discriminator = SNDCGAN_Discrminator(**config)
data_set = Cifar10(batch_size=FLAGS.batch_size)

global_step = tf.Variable(0, name="global_step", trainable=False)
increase_global_step = global_step.assign(global_step + 1)
is_training = tf.placeholder(tf.bool, shape=())
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_hat = generator(z, is_training=is_training)
x = tf.placeholder(tf.float32, shape=x_hat.shape)

d_fake = discriminator(x_hat, update_collection=SPECTRAL_NORM_UPDATE_OPS)
# Don't need to collect on the second call, put NO_OPS
d_real = discriminator(x, update_collection="NO_OPS")
# Softplus at the end as in the official code of author at chainer-gan-lib github repository
d_loss = tf.reduce_mean(tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real))
g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake))
d_loss_summary_op = tf.summary.scalar('d_loss', d_loss)
g_loss_summary_op = tf.summary.scalar('g_loss', g_loss)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('snapshots')

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
d_gvs = optimizer.compute_gradients(d_loss, var_list=d_vars)
g_gvs = optimizer.compute_gradients(g_loss, var_list=g_vars)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
d_solver = optimizer.apply_gradients(d_gvs)
with tf.control_dependencies(update_ops):
  g_solver = optimizer.apply_gradients(g_gvs)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if tf.train.latest_checkpoint('snapshots') is not None:
  saver.restore(sess, tf.train.latest_checkpoint('snapshots'))

np.random.seed(1337)
sample_noise = generator.generate_noise()
np.random.seed()
iteration = sess.run(global_step)
start = timeit.default_timer()
spectral_norm_update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS)

is_start_iteration = True
inception_scores = []
while iteration < FLAGS.max_iter:
  _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z: generator.generate_noise(), is_training: True})
  for _ in range(FLAGS.n_dis):
    _, d_loss_curr, summaries = sess.run([d_solver, d_loss, merged_summary_op],
                                         feed_dict={x: data_set.get_next_batch(), z: generator.generate_noise(), is_training: True})
  if (iteration + 1) % FLAGS.display_interval == 0 and not is_start_iteration:
    summary_writer.add_summary(summaries, global_step=iteration)
    stop = timeit.default_timer()
    print('Iter {}: d_loss = {:4f}, g_loss = {:4f}, time = {:2f}s'.format(iteration, d_loss_curr, g_loss_curr, stop - start))
    start = stop
  if (iteration + 1) % FLAGS.snapshot_interval == 0 and not is_start_iteration:
    saver.save(sess, 'snapshots/model.ckpt', global_step=iteration)
    sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'tmp/{:06d}.png'.format(iteration))
  if (iteration + 1) % FLAGS.evaluation_interval == 0:
    sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'tmp/{:06d}.png'.format(iteration))
    # Sample 50000 images for evaluation
    print("Evaluating...")
    num_images_to_eval = 50000
    eval_images = []
    num_batches = num_images_to_eval // FLAGS.batch_size + 1
    print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
    np.random.seed(0)
    for _ in range(num_batches):
      images = sess.run(x_hat, feed_dict={z: generator.generate_noise(), is_training: False})
      eval_images.append(images)
    np.random.seed()
    eval_images = np.vstack(eval_images)
    eval_images = eval_images[:num_images_to_eval]
    eval_images = np.clip((eval_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    # Calc Inception score
    eval_images = list(eval_images)
    inception_score_mean, inception_score_std = get_inception_score(eval_images)
    print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))
    inception_scores.append(dict(mean=inception_score_mean, std=inception_score_std))
    with open(INCEPTION_FILENAME, 'wb') as f:
      pickle.dump(inception_scores, f)
  # Update Spectral Norm left vectors
  for update_op in spectral_norm_update_ops:
    sess.run(update_op)
  iteration += 1
  sess.run(increase_global_step)
  is_start_iteration = False
