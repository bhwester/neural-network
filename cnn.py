# Computer Vision final project
# Author: Brian Westerman

"""
Implement a convolutional neural network in the machine learning Python API TensorFlow.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # holds pointer to MNIST data


# Define some functions that make code more concise and modular so we don't have type out 
# TensorFlow operations a bunch of times

# Initialize weights in a Variable tensor
def weight(shape):
	init = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
	return tf.Variable(initial_value=init)

# Initialize biases in a Variable tensor
def bias(shape):
	init = tf.constant(value=0.1, shape=shape)
	return tf.Variable(initial_value=init)

# Create an Operation for convolution
def convolve(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Create an Operation for 2x2 max pooling
def maxpool(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Build the computational graph in a TensorFlow Session (the context manager)
sess = tf.Session()

# Weights and biases for convolutional layer 1
W_conv1 = weight([5, 5, 1, 32])  # (800 weights)
b_conv1 = weight([32])

# Create a Placeholder tensor for the input data and true output labels
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_label = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# Convolution and pooling Operation for convolutional layer 1
h_conv1 = tf.nn.relu(convolve(x_image, W_conv1) + b_conv1)  # 28x28x1 -> 28x28x32
h_pool1 = maxpool(h_conv1)  # 28x28x32 -> 14x14x32

# Weights and biases for convolutional layer 2
W_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])

# Convolution and pooling Operation for convolutional layer 2
h_conv2 = tf.nn.relu(convolve(h_pool1, W_conv2) + b_conv2)  # 14x14x32 -> 14x14x64
h_pool2 = maxpool(h_conv2)  # 14x14x64 -> 7x7x64

# Weights and biases for fully connected layer 1
W_fc1 = weight([7*7*64, 1024])
b_fc1 = bias([1024])

# Activation function for fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # 7*7*64 = 3,136 neurons flattened
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 3,136 -> 1,024 (3,211,264 weights)

# Implement dropout, TensorFlow takes care of the details in the computational graph
keep_probability = tf.placeholder(dtype=tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_probability)

# Weights and biases for fully connected layer 2
W_fc2 = weight([1024, 10])
b_fc2 = bias([10])

# Predicted output
y_prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 1024 -> 10 (10,240 weights)

# Build out the final steps of the computational graph so the model can be automatically 
# trained via backpropagation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y_prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Prepare the Session to be run by initializing all Variables
sess.run(tf.initialize_all_variables())

# Train the model
for i in range(20000):
	batch = mnist.train.next_batch(50)
	# Print train accuracy every 100 iterations
	if i % 100 == 0:
		train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], 
									y_label: batch[1], 
									keep_probability: 1.0})
		print("Step %d, training accuracy %g"%(i, train_accuracy))
	# Run one epoch of training with dropout set to 50% keep probability
	train_step.run(session=sess, feed_dict={x: batch[0], 
						y_label: batch[1], 
						keep_probability: 0.5})

# Print test accuracy (TensorFlow automatically partitions train and test data)
print("Test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist.test.images, 
								y_label: mnist.test.labels, 
								keep_probability: 1.0}))
