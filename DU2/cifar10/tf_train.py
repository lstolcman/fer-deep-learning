import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import math
import pickle
import numpy as np
import skimage as ski
import skimage.io

def init_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
        
def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


DATA_DIR = '/home/marko/Projects/datasets/cifar-10-batches-py/'
SAVE_DIR = "/home/marko/Projects/source/fer/cifar10/"
init_dir(SAVE_DIR)

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-4
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

# DATA
img_height, img_width, num_channels = 32, 32, 3
train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']

train_x = train_x.reshape((-1, img_height, img_width, num_channels))
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, img_height, img_width, num_channels)).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0,1,2))
data_std = train_x.std((0,1,2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std
print(train_x.shape)

weight_decay = config['weight_decay']

n_classes = 10

def init_var(shape, fin):
    sigma = np.sqrt(2/fin)
    return tf.Variable(tf.truncated_normal(shape, stddev=sigma))

def l2_loss(weights):
    regularizers = 0;
    for w in weights:
        regularizers += tf.nn.l2_loss(w)
    return regularizers
        

def conv2d(x, W, b, activation=tf.nn.relu, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return activation(x)

def maxpool2d(x, k=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')

def fc(x, W, b, activation=None):
    x = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
    if activation :
        return activation(tf.matmul(x, W) +  b)    
    return tf.matmul(x, W) +  b



weights = {
    'conv1': init_var([5, 5, 3, 16], 5*5*3),  # 5x5 conv, 3 input, 16 outputs
    'conv2': init_var([5, 5, 16, 32], 5*5*16), # 5x5 conv, 16 inputs, 32 outputs
    
    'fc1': init_var([8*8*32, 256], 8*8*32), # fully connected, 8*8*32 inputs, 256 outputs
    'fc2': init_var([256, 128], 256), # fully connected, 256 inputs, 128 outputs
    'fc3': init_var([128, n_classes], 128) # 128 inputs, 10 outputs (class prediction)
}

biases = {
    'conv1': tf.Variable(tf.zeros([16])),
    'conv2': tf.Variable(tf.zeros([32])),
    'fc1': tf.Variable(tf.zeros([256])),
    'fc2': tf.Variable(tf.zeros([128])),
    'fc3': tf.Variable(tf.zeros([n_classes]))
}

#conv(16,5) -> pool(3,2) -> conv(32,5) -> pool(3,2) -> fc(256) -> fc(128) -> fc(10)
def convnet(x, weights, biases):
    x = tf.reshape(x, shape=[-1, img_height, img_width, num_channels])
    net = conv2d(x, weights['conv1'], biases['conv1'], tf.nn.relu)
    net = maxpool2d(net, k=3, stride=2)
    
    net = conv2d(net, weights['conv2'], biases['conv2'], tf.nn.relu)
    net = maxpool2d(net, k=3, stride=2)
    
    net = fc(net, weights['fc1'],  biases['fc1'], tf.nn.relu)
    net = fc(net, weights['fc2'],  biases['fc2'], tf.nn.relu)
    net = fc(net, weights['fc3'],  biases['fc3'])
    return net



# Graph
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y_ = tf.placeholder(tf.int32, [None,])
logits = convnet(X, weights, biases)

# loss
regularizers = l2_loss(weights.values())
data_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, Y_))
loss = data_loss + weight_decay*regularizers

lr = tf.placeholder(tf.float32)
train_step =  tf.train.GradientDescentOptimizer(lr).minimize(loss)


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
      batch_x = train_x[i*batch_size:(i+1)*batch_size, ...]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, ...]
    
      data_dict = {X: batch_x, Y_: batch_y, lr:solver_config['lr']}
      logits_val, loss_val, _ = session.run([logits, loss, train_step] ,feed_dict=data_dict)

    
      # compute classification accuracy
      yp = np.argmax(logits_val, 1)
      yt = batch_y
      cnt_correct += (yp == yt).sum()


      if i % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))

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
    
    data_dict = {X: batch_x, Y_: batch_y}
    logits_val, loss_val = session.run([logits, loss] ,feed_dict=data_dict)
    
    yp = np.argmax(logits_val, 1)
    yt = batch_y
    cnt_correct += (yp == yt).sum()

    loss_avg += loss_val
  valid_acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches
  print(name + " accuracy = %.2f" % valid_acc)
  print(name + " avg loss = %.2f\n" % loss_avg)

    
session = tf.Session()
train(session, train_x, train_y, valid_x, valid_y, config)