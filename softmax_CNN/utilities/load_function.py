import numpy as np
import pickle

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    x = dict[b'data']
    y = dict[b'labels']
    x = x.astype(float)
    y = np.array(y)
  return x, y

def load_data():

  xs = []
  ys = []
  # for i in range(1, 6):
  #   filename = 'data/cifar/data_batch_' + str(i)
  #   X, Y = unpickle(filename)
  #   xs.append(X)
  #   ys.append(Y)
  filename = 'data/cifar/data_batch_1'
  X, Y = unpickle(filename)
  xs.append(X)
  ys.append(Y)
  x_train = np.concatenate(xs)
  y_train = np.concatenate(ys)
  del xs, ys

  x_test, y_test = unpickle('data/cifar/data_batch_1')

  data_dict = {
    'images_train': x_train,
    'labels_train': y_train,
    'images_test': x_test,
    'labels_test': y_test,
  }
  return data_dict