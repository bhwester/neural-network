__author__ = 'Brian'
# My first neural net!!!

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Consider implementing feature scaling/whitening (scipy.cluster.vq.whiten?)
# Consider implementing PCA whitening
# Consider implementing an autoencoder
# Consider implementing other optimization algorithms besides vanilla gradient descent, such as stochastic gradient descent,
# Adagrad, Adadelta, Adam, Nesterov's accelerated gradient descent, momentum, RMSprop
# Involve learning rate decay?
# Consider implementing dropout and maxout
# Consider implementing other activation functions (any others?)
# Consider implementing k-fold cross-validation and confusion matrix for classification to validate model performance
# Consider implementing a RNN
# Consider implementing a Reinforcement Learning agent
# Consider implementing a genetic algorithm or other evolutionary algorithms
# Consider implementing a Hidden Markov Model
# Consider implementing a SVM
# Consider implementing a SOM
# Consider implementing Attention Mechanisms
# Consider using deep learning frameworks like TensorFlow, Theano, Caffe, Torch, Neon, Keras, etc.
# Consider making a model with SyntaxNet


# Sigmoid function to get "activations" in [0, 1] for nodes in hidden layer:
# g(z) = 1/(1+e^(-z))
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Tanh function to get "activations" in [-1, 1] for nodes in the hidden layer:
# g(z) = 2/(1+e^(-2z)) - 1
def tanh(z):
    return 2/(1 + np.exp(-2*z)) - 1

# Computes leaky ReLU ( max(0, z) ) (normal RelU uses alpha = 0)
def relu(z):
    alpha = 0.01  # can be modified
    if z < 0:
        return alpha * z
    else:
        return z

# Softmax function to get "activations" in [, ] for nodes in the hidden layer:
# P(y=k|x;theta) = e^(thetak*x)/sumK(e^(theta*x)) where k in {1, 2,..., K}
# g(z) = e^z[k]/sum(e^z)
def softmax(z, k):
    return np.exp(z[k-1])/np.sum(np.exp(z))

# Softplus function to get "activations" ( "softer" RelU, which is max(0, z) )
# g(z) = log(1+e^z)
# derivative of softplus is simply the sigmoid function
def softplus(z):
    return np.log(1 + np.exp(z))

# Derivative of sigmoid function to compute gradient terms in the hidden layer:
# g'(z) = sigmoid(z)*(1-sigmoid(z)) for sigmoid function
def dsigmoid(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

# Derivative of tanh function to compute gradient terms in the hidden layer:
# g'(z) = (1+tanh(z))*(1-tanh(z)) for tanh function
def dtanh(z):
    return np.multiply((1 + tanh(z)), (1 - tanh(z)))

# Derivative of ReLU
def drelu(z):
    alpha = 0.01
    if z < 0:
        return alpha
    else:
        return 1

# Calculate error term of hidden layer:
# # error2 = (theta2.T*error3) .* g'(z2)
def calcErrorTerm(theta, error, z):
    return np.multiply((theta[:, 1:].T * error), dtanh(z))

# Calculate the regularized cost function for logistic regression:
# J(theta) = (1/m)*sum(-y*log(h)-(1-y)*log(1-h)) + (lambda/2m)*(sum(theta1^2)+sum(theta2^2))
def calcCostLg(h, y, theta1, theta2):
    m = y.shape[0]
    cost = 0
    cost += np.sum(-np.multiply(y, np.log10(h)) - np.multiply((1 - y), np.log10(1 - h))) \
            + (regLambda/(2*m)) * (np.sum(np.square(theta1)) + np.sum(np.square(theta2)))
    return cost

# Calculate the regularized cost function for linear regression:
# J(theta) = (1/2)*(sum(h - y)^2 + lambda*(sum(theta1^2)+sum(theta2^2))
def calcCostLr(h, y, theta1, theta2):
    m = y.shape[0]
    J = 1/2 * (np.sum(np.square(h - y)) + (regLambda * (np.sum(np.dot(theta1.T, theta1)) + np.sum(np.dot(theta2.T, theta2)))))
    return J


"""
Multilayer perceptron
"""
# Train the neural net
def trainPerceptron():

    # Read in data
    filename = "file"
    data = pd.read_csv(filename)
    input = data[:, :-4]
    y = data[:, -4:]

    # Initialize key values
    m = input.shape[0]
    j1 = input.shape[1] + 1
    j2 = 6
    j3 = 4
    epsilon = 0.13
    numLayers = 3
    targetCost = 0.0001
    cost = 99999999
    alpha = 0.01
    regLambda = 1

    # Initialize weights
    theta1 = np.random.rand(j2-1, j1) * (2*epsilon) - epsilon
    theta2 = np.random.rand(j3, j2) * (2*epsilon) - epsilon

    while (cost >= targetCost):
    # for j in range(1000):
        # initialize a matrix to store the predictions
        h = np.zeros((m, j3))
        # initialize a count to accumulate adjustments to the weights
        gradient1 = np.zeros((j2, j1+1))
        gradient2 = np.zeros((j3, j2+1))
        # Determine delta matrix for each layer
        for i in range(m):
            # Forward propagation
            a1 = input[i].T
            a1 = np.vstack((np.ones((1, 1)), a1))
            z2 = np.dot(theta1, a1b)
            a2 = tanh(z2)
            a2 = np.vstack((np.ones((1, 1)), a2))
            z3 = np.dot(theta2, a2b)
            a3 = tanh(z3)
            h[i, :] = a3
            # Backpropagation
            actual = y[i].T
            delta3 = a3 - actual
            delta2 = calcErrorTerm(theta2, error3, z2)
            # Calculate adjustments for weights for this iteration
            adjustments1 = np.dot(delta2, a1.T)  # careful, bias term doesn't get multiplied through
            adjustments2 = np.dot(delta3, a2.T)  # careful, bias term doesn't get multiplied through
            # Accumulate adjustments
            gradient1 += adjustments1
            gradient2 += adjustments2
        # Adjust weights using regularization
        adjustBias = alpha * (gradient1[:, 0] / m)
        adjustWeights = alpha * (gradient1[:, 1:] / m + ((regLambda/m) * theta1[:, 1:]))
        theta1[:, 0] -= adjustBias
        theta1[:, 1:] -= adjustWeights
        adjustBias = alpha * (gradient2[:, 0] / m)
        adjustWeights = alpha * (gradient2[:, 1:] / m + ((regLambda/m) * theta2[:, 1:]))
        theta2[:, 0] -= adjustBias
        theta2[:, 1:] -= adjustWeights
        cost = calcCostLg(h, y, theta1, theta2)



"""
Convolutional neural network (LeNet)
"""
# It may be a lot easier to learn something like Theano or TensorFlow and use it for functions like convolution and pooling
# Flatten image into a one-dimensional vector to reduce dimensions of tensors by one?
# Does deconvolution actually need to be implemented by dividing the fourier transforms of delta by W then taking the inverse fourier transform?
##-> means that the corresponding operation is run here, likely using a machine learning library
def trainCNN():

    images = []
    images.append("all images in np.matrix form")
    y = ["correct labels"]

    alpha = 0.01
    regLambda = 1
    epsilon = 0.13
    channels = 3  # RGB or grayscale
    kernelSize = (5, 5)  # size of convolution kernel (could be different for various layers depending on image size)
    maxPool = (2, 2)  # stride of subsampling pool (could be different for various layers, and could be mean or L^p pooling)
    imageShape = images[0].shape  # dimensions of input images (assume 32x32)
    c1 = 4  # number of convolved feature maps in layer 1
    s1 = c1  # number of pooled feature maps in layer 1
    c2 = 12  # number of convolved feature maps in layer 2
    s2 = c2  # number of pooled feature maps in layer 2
    n1 = 20  # number of nodes in fully connected layer 1 (there could be more hidden layers)
    n2 = 10  # number of nodes in fully connected layer 2 (output layer)

    W1 = np.random.rand(c1, 1, kernelSize[0], kernelSize[1], channels) * (2*epsilon) - epsilon  # numpy array of convolution kernels connecting input image to c1
    b1 = np.random.rand(c1, 1) * (2*epsilon) - epsilon  # biases for convolution kernels connecting input image to c1
    W2 = np.random.rand(c2, s1, kernelSize[0], kernelSize[1], channels) * (2*epsilon) - epsilon  # numpy array of convolution kernels connecting s1 to c2
    b2 = np.random.rand(c2, s1) * (2*epsilon) - epsilon  # biases for convolution kernels connecting s1 to c2
    W3 = np.random.rand(n1, s2, kernelSize[0], kernelSize[1], channels) * (2*epsilon) - epsilon  # numpy array of convolution kernels connecting s2 to n1
    b3 = np.random.rand(n1, s2) * (2*epsilon) - epsilon  # biases for convolution kernels connecting s2 to n1
    W4 = np.random.rand(n2, n1) * (2*epsilon) - epsilon  # weights connecting n1 to n2
    b4 = np.random.rand(n2) * (2*epsilon) - epsilon  # weights for n1 bias term

    for p in range(len(images)):

        # Is there a better way to vectorize all this?
        # Reshape dimensions of tensors to be consistent with TensorFlow?

        image = images[p]  # should be (32, 32, 3)
        c1Convolved = np.zeros((c1, imageShape[0]-kernelSize[0]+1, imageShape[1]-kernelSize[1]+1, channels))  # should be (4, 28, 28, 3)
        c1Activated = np.zeros(c1Convolved.shape)  # should be (4, 28, 28, 3)
        c1Pooled = np.zeros((c1Convolved.shape[0], c1Convolved.shape[1]/maxPool[0], c1Convolved.shape[2]/maxPool[1], channels))  # should be (4, 14, 14, 3)
        c2Convolved = np.zeros((c2, c1Pooled.shape[0]-kernelSize[0]+1, c1Pooled.shape[1]-kernelSize[1]+1, channels))  # should be (12, 10, 10, 3)
        c2Activated = np.zeros(c2Convolved.shape)  # should be (12, 10, 10, 3)
        c2Pooled = np.zeros((c2Convolved.shape[0], c2Convolved.shape[1]/maxPool[0], c2Convolved.shape[2]/maxPool[1], channels))  # should be (12, 5, 5, 3)
        n1Convolved = np.zeros((n1))
        n1Activated = np.zeros((n1))
        n2Convolved = np.zeros((n2))
        n2Activated = np.zeros((n2))
        delta1Convolved = np.zeros(c1Convolved.shape)  # should be (4, 28, 28, 3)
        delta1Pooled = np.zeros(c1Pooled.shape)  # should be (4, 14, 14, 3)
        delta2Convolved = np.zeros(c2Convolved.shape)  # should be (12, 10, 10, 3)
        delta2Pooled = np.zeros(c2Pooled.shape)  # should be (12, 5, 5, 3)
        delta3 = np.zeros(n1)
        delta4 = np.zeros(n2)
        # initialize an array to store predictions
        h = np.zeros((n2))

        # Forward propagation layer 1
        for i in range(c1):
            ##-> convolve image with W1[i, 0, :, :, :], add b1[i, 0], and store it in c1Convolved[i, :, :, :]

        ##-> run activation function on c1Convolved[:, :, :, :] for each pixel and channel, and store it in c1Activated[:, :, :, :]
        ##-> run max pooling on c1Activated[:, :, :, :] and store it in c1Pooled[:, :, :, :]

        # Forward propagation layer 2
        for i in range(c2):
            for j in range(c1):
                ##-> convolve c1Pooled[j, :, :, :] with W2[i, j, :, :, :], add b2[i, j], and add it to c2Convolved[i, :, :, :]

        ## run activation function on c2Convolved[:, :, :, :] for each pixel and channel, and store it in c2Activated[:, :, :, :]
        ## run max pooling on c2Activated[:, :, :, :] and store it in c2Pooled[:, :, :, :]

        # Forward propagation layer 3
        for i in range(n1):
            for j in range(c2):
                ##-> convolve c2Pooled[j, :, :, :] with W3[i, j, :, :, :], add b3[i, j], average the channels (yes?), and add the resulting number to n1Convolved[i]

        ##-> run activation function on n1Convolved and store it in n1Activated

        # Forward propagation layer 4
        n2Convolved += np.dot(W4, n1Activated)
        n2Convolved += b4
        ##-> run softmax activation function on n2Convolved and store it n2Activated

        # Backpropagation layer 4
        delta4 = n2Activated - y

        # Backpropagation layer 3
        delta3 = calcErrorTerm(W4, delta4, n1Convolved)  # don't need to factor in b4 to calculating delta3

        # Backpropagation layer 2
        for i in range(c2):
            for j in range(n1):
                ##-> deconvolve delta3[j] with W3[j, i, :, :, :] and add it to delta2Pooled[i, :, :, :]
                        # expands shape to that of delta2Pooled, and means error is being distributed through all (3) channels

        ##-> upsample delta2Pooled[:, :, :, :] and store it in delta2Convolved[:, :, :, :]
        ##-> multiply element-wise delta2Convolved[:, :, :, :] with the result of running c2Convolved[:, :, :, :]
                # through the derivative of the activation function and store it in delta2Convolved[:, :, :, :]

        # Backpropagation layer 1
        for i in range(c1):
            for j in range(c2):
                ##-> deconvolve delta2Convolved[j, :, :, :] with W2[j, i, :, :, :] and add it to delta1Pooled[i, :, :, :]
                        # expands shape to that of delta1Pooled, and means error is continuing to be distributed through all (3) channels

        ##-> upsample delta1Pooled[:, :, :, :] and store it in delta1Convolved[:, :, :, :]
        ##-> multiply element-wise delta1Convolved[:, :, :, :] with the result of running c1Convolved[:, :, :, :]
                # through the derivative of the activation function and store it in delta1Convolved[:, :, :, :]

        # Compute gradients for layer 1
        for i in range(c1):
            ##-> convolve image with delta1Convolved[i, :, :, :] and subtract that (times alpha) from W1[i, 0, :, :, :]
            ##-> average three channels of delta1Convolved[i, :, :, :] and subtract the width and height dimensions from b1[i, 0]
            # TODO: Regularization

        # Compute gradients for layer 2
        for i in range(c2):
            for j in range(c1):
                ##-> convolve c1Pooled[j, :, :, :] with delta2Convolved[i, :, :, :] and subtract that (times alpha) from W2[i, j, :, :, :]
                ##-> average three channels of delta2Convolved[i, :, :, :] and subtract the width and height dimensions from b2[i, j]
                # TODO: Regularization

        # Compute gradients for layer 3
        for i in range(n1):
            for j in range(c2):
                ##-> convolve c2Pooled[j, :, :, :] with delta3[i] and subtract that (times alpha) from W3[i, j, :, :, :]
                ##-> subtract delta3[i] from b3[i, j]
                # TODO: Regularization

        # Compute gradients for layer 4
        W4 -= alpha * (np.outer(delta4, n1Activated) + (regLambda * W4))  # is regLambda correct? What about m?
        b4 -= delta4

        # FIXME: Fix biases, right now their operations don't make sense at all
        # Biases should have one vector component for each output node of a given layer



"""
Implement a convolutional neural network in the machine learning Python API TensorFlow.
"""

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
