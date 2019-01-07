# External Dependencies
import numpy as np
import tensorflow as tf
import os

def softmax_cnn(data_sets,classes, settings):

  s = settings
  ##__Set Up TF Graph
  # Input Placeholders
  x = tf.placeholder(tf.float32, shape=[None, 32 * 32 * 3])
  y = tf.placeholder(tf.int64, shape=[None])

  # Weights+Biases for convolutional filters
  c1_weights = tf.Variable(tf.random_normal([s['filter_dim'], s['filter_dim'], 3, s['filters']]))
  c1_biases = tf.Variable(tf.constant(.1, shape=[s['filters']]))

  # Weights+Biases for fully connected hidden layer
  h_fc_weights = tf.Variable(tf.random_normal([16 * 16 * s['filters'], s['dense_h_fc_units']]))
  h_fc_biases = tf.Variable(tf.random_normal([s['dense_h_fc_units']]))

  # Weights+Biases for output channel
  softmax_weights = tf.Variable(tf.truncated_normal([s['dense_h_fc_units'], len(classes)], stddev=0.01))
  softmax_biases = tf.Variable(tf.constant(0.1, shape=[len(classes)]))

  # reshape x for the convolutional layer
  x_image = tf.reshape(x, [-1, 32, 32, 3])

  # convolutional layer (outputs the covolved image filters)
  conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(x_image, c1_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv_layer'), c1_biases))

  # pooling layer (outputs pooled filters) (but could theoretically be applied to any image)
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # fully connected hidden layer
  h_fc_input = tf.reshape(pool1, [-1, 16 * 16 * s['filters']])
  h_fc_activation = tf.nn.relu(tf.add(tf.matmul(h_fc_input, h_fc_weights), h_fc_biases))

  # hidden outputs with dropout trick
  keep = tf.placeholder(tf.float32)  # how much
  h_fc_activation_drop = tf.nn.dropout(h_fc_activation, keep)

  # output layers, cost functions, optimizers
  logits = tf.add(tf.matmul(h_fc_activation_drop, softmax_weights), softmax_biases)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y))
  train_step = tf.train.RMSPropOptimizer(s['learning_rate']).minimize(loss)

  correct_prediction = tf.equal(tf.argmax(logits, 1), y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  ##__Run Model
  # _
  # information storage constainers
  batch_size = 100

  # run tensorflow (graph) session
  with tf.Session() as sess:
    # initialize all the variables into the tesnorflow environment
    sess.run(tf.global_variables_initializer())
    for i in range(s['epochs']):

      indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
      images_batch = data_sets['images_train'][indices]
      labels_batch = data_sets['labels_train'][indices]

      _, cost_val = sess.run([train_step, loss], feed_dict={x: images_batch, y: labels_batch, keep: s['dropout_ratio']})
      print('Cost: ', cost_val)


      test_accuracy = sess.run(accuracy, feed_dict={x: data_sets['images_test'], y: data_sets['labels_test'], keep: s['dropout_ratio']})
      print('Test accuracy {:g}'.format(test_accuracy))


