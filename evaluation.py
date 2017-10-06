import numpy as np
import tensorflow as tf
import sys

tf.reset_default_graph()

# he initializer
he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0)
l2_reg = tf.contrib.layers.l2_regularizer(0.001)
# l2_reg = None

# make model
X = tf.placeholder(tf.float32,shape=[None,50,50,50,1])
y = tf.placeholder(tf.int32, shape=[None])

conv1 = tf.layers.conv3d(X,12,[5,5,5], padding='same', kernel_initializer=he_init, kernel_regularizer=l2_reg)
relu1 = tf.nn.elu(conv1)
maxpool1 = tf.layers.max_pooling3d(relu1, [4,4,4],[3,3,3])

conv2 = tf.layers.conv3d(maxpool1, 24, [5,5,5], padding='same', kernel_initializer=he_init, kernel_regularizer=l2_reg)
relu2 = tf.nn.elu(conv2)
maxpool2 = tf.layers.max_pooling3d(relu2, [4,4,4], [3,3,3])

n_layer = np.prod(maxpool2.shape[1:])._value
flat_layer = tf.reshape(maxpool2,shape=[-1,n_layer])

dense1 = tf.layers.dense(flat_layer, 1500, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_reg)
dense2 = tf.layers.dense(dense1, 1000, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_reg)
logits = tf.layers.dense(dense2,19,kernel_initializer=he_init)

# loss function
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train_op = optimizer.minimize(loss)

# accuracy evaluation
correct_predictions = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

# load data
from csp_tools import *
#X_train, y_train = load_data_range(range(19),range(40))
#X_val, y_val = load_data_range(range(19),range(40,45))
X_test, y_test = load_data_range(range(19), range(45,50))

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'CSP.epoch_99.ckpt')
    acc = sess.run(accuracy, feed_dict = {X: X_test, y: y_test})
    print('test set accuracy =',acc)



