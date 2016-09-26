##
## Python 3
##
## Code from tutorial at:
## https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html#deep-mnist-for-experts
##
## NB:  You have to wait a bit the first time you run this, since it needs to
##		download the MNIST dataset (ca. 12 MB)
##

# adjust keep_rate to modify dropout. keep_rate = 1.0 gives no dropout
keep_rate = 0.5
# use 20 000 iterations for good results (ca 99.2%). But that may take up to half an hour to run.
num_training_iterations = 500

# size of the MNIST test set
test_size   = 10000
# just in case
hurt_humans = False


print("Keep rate:", keep_rate)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

# input values
x = tf.placeholder(tf.float32, shape=[None, 784])
# target values
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densly connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# add dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# measure runtime
import timeit
start = timeit.default_timer()

# train and evaluate
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(num_training_iterations):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))

	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: keep_rate})
	
# measure accuracy on test set
print("starting test...")

# modified version that uses batches. The original version needs 6-7 GB RAM to run.
acc = 0
iterations = test_size/10
for i in range(int(iterations)):
    batch = mnist.test.next_batch(10)
    acc = acc + accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})

# print accuracy
acc = acc/iterations
print("test accuracy:", acc)

# original version. use this version if you want to use all your RAM
#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images,
#							y_: mnist.test.labels, keep_prob: 1.0}))

# print runtime
stop = timeit.default_timer()
print("Runtime:", stop - start)

sess.close()