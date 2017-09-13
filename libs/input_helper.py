from keras.datasets import cifar10
import numpy as np

class Cifar10(object):
  def __init__(self, batch_size=64, test=False):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if test:
      images = x_test
    else:
      images = x_train
    self.images = (images - 127.5) / 127.5
    self.batch_size = batch_size
    self.num_samples = len(self.images)
    self.shuffle_samples()
    self.next_batch_pointer = 0

  def shuffle_samples(self):
    image_indices = np.random.permutation(np.arange(self.num_samples))
    self.images = self.images[image_indices]

  def get_next_batch(self):
    num_samples_left = self.num_samples - self.next_batch_pointer
    if num_samples_left >= self.batch_size:
      batch = self.images[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
      self.next_batch_pointer += self.batch_size
    else:
      partial_batch_1 = self.images[self.next_batch_pointer:self.num_samples]
      self.shuffle_samples()
      partial_batch_2 = self.images[0:self.batch_size - num_samples_left]
      batch = np.vstack((partial_batch_1, partial_batch_2))
      self.next_batch_pointer = self.batch_size - num_samples_left
    return batch
