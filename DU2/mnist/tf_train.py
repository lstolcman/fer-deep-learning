import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import nn
import os
import math

import numpy as np
import skimage as ski
import skimage.io

def init_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

DATA_DIR = '/home/marko/Projects/datasets/MNIST/'
SAVE_DIR = "/home/marko/Projects/source/fer/contrib_out_l2_1e3/"
init_dir(SAVE_DIR)

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-4
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

# DATA
np.random.seed(int(time.time() * 1e6) % 2**31)
dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)

train_x = dataset.train.images
train_x = train_x.reshape([-1, 28, 28, 1])
train_y = dataset.train.labels

valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 28, 28, 1])
valid_y = dataset.validation.labels

test_x = dataset.test.images
test_x = test_x.reshape([-1, 28, 28, 1])
test_y = dataset.test.labels

train_mean = train_x.mean()
train_x -= train_mean
valid_x -= train_mean
test_x -= train_mean

weight_decay = config['weight_decay']

n_input = 768
n_classes = 10

def init_var(shape, fin):
    sigma = np.sqrt(2/fin)
    return tf.Variable(tf.truncated_normal(shape, stddev=sigma))



weights = {
    'conv1': init_var([5, 5, 1, 16], 5*5),  # 5x5 conv, 1 input, 16 outputs
    'conv2': init_var([5, 5, 16, 32], 5*5*16), # 5x5 conv, 16 inputs, 32 outputs
    
    'fc1': init_var([7*7*32, 512], 7*7*32), # fully connected, 7*7*32 inputs, 512 outputs
    'fc2': init_var([512, n_classes], 512) # 512 inputs, 10 outputs (class prediction)
}

biases = {
    'conv1': tf.Variable(tf.zeros([16])),
    'conv2': tf.Variable(tf.zeros([32])),
    'fc1': tf.Variable(tf.zeros([512])),
    'fc2': tf.Variable(tf.zeros([n_classes]))
}

def l2_loss(weights):
    regularizers = 0;
    for w in weights:
        regularizers += tf.nn.l2_loss(w)
    return regularizers
        

def conv2d(x, W, b, activation=tf.nn.relu, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return activation(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def fc(x, W, b, activation=None):
    x = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
    if activation :
        return activation(tf.matmul(x, W) +  b)    
    return tf.matmul(x, W) +  b

def convnet(x, weights, biases):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    net = conv2d(x, weights['conv1'], biases['conv1'], tf.nn.relu)
    net = maxpool2d(net, k=2)
    
    net = conv2d(net, weights['conv2'], biases['conv2'], tf.nn.relu)
    net = maxpool2d(net, k=2)
    
    net = fc(net, weights['fc1'],  biases['fc1'], tf.nn.relu)
    net = fc(net, weights['fc2'],  biases['fc2'])
    return net



# Graph

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Yoh_ = tf.placeholder(tf.float32, [None, n_classes])
logits = convnet(X, weights, biases)

# loss
regularizers = l2_loss([weights['conv1'], weights['conv2'], weights['fc1']])
data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, Yoh_))
loss = data_loss + weight_decay*regularizers

lr = tf.placeholder(tf.float32)
train_step =  tf.train.GradientDescentOptimizer(lr).minimize(loss)

def draw_conv_filters(epoch, step, name, weights, save_dir):
  # kxkxCxn_filters
  k, k, C, num_filters = weights.shape

  w = weights.copy().swapaxes(0, 3).swapaxes(1,2)
  w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()

  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border

  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (name, epoch, step, i)
    ski.io.imsave(os.path.join(save_dir, filename), img)

def train(session, train_x, train_y, valid_x, valid_y, config):
  session.run(tf.initialize_all_variables())

  lr_policy = config['lr_policy']
  batch_size = config['batch_size']
  max_epochs = config['max_epochs']
  save_dir = config['save_dir']
  num_examples = train_x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
    
    
  for epoch in range(1, max_epochs+1):
    if epoch in lr_policy:
      solver_config = lr_policy[epoch]
    cnt_correct = 0

    permutation_idx = np.random.permutation(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]

    for i in range(num_batches):
      # store mini-batch to ndarray
      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
    
      data_dict = {X: batch_x, Yoh_: batch_y, lr:solver_config['lr']}
      logits_val, loss_val, _ = session.run([logits, loss, train_step] ,feed_dict=data_dict)

    
      # compute classification accuracy
      yp = np.argmax(logits_val, 1)
      yt = np.argmax(batch_y, 1)
      cnt_correct += (yp == yt).sum()


      if i % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))
      if i % 100 == 0:
        w = session.run(weights['conv1'])
        draw_conv_filters(epoch, i*batch_size, "conv1", w, save_dir)
      if i > 0 and i % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
        
    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
    evaluate(session, "Validation", valid_x, valid_y, config)
  return net


def evaluate(session, name, x, y, config):
  print("\nRunning evaluation: ", name)
  batch_size = config['batch_size']
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  cnt_correct = 0
  loss_avg = 0


  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    
    data_dict = {X: batch_x, Yoh_: batch_y}
    logits_val, loss_val = session.run([logits, loss] ,feed_dict=data_dict)
    
    yp = np.argmax(logits_val, 1)
    yt = np.argmax(batch_y, 1)
    cnt_correct += (yp == yt).sum()

    loss_avg += loss_val
  valid_acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches
  print(name + " accuracy = %.2f" % valid_acc)
  print(name + " avg loss = %.2f\n" % loss_avg)

    
session = tf.Session()
train(session, train_x, train_y, valid_x, valid_y, config)
evaluate(session, "Test", test_x, test_y, config)