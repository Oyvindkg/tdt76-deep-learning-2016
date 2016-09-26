import tensorflow as tf


# Number of nodes in hidden layer
number_of_hidden_nodes = 20


# Input values and corresponding output values
# Defined as floating number values instead of integers to avoid having
# to cast them to integers when multiplying them with the weights
x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]


# Weights and biases between input layer and hidden layer
# Weights are uniformly chosen in the range [-0.1, 0.1)
# Biases are uniformly chosen in the range [0.0, 0.01)
W = tf.Variable(tf.random_uniform([2, number_of_hidden_nodes], -0.1, 0.1))
b = tf.Variable(tf.random_uniform([number_of_hidden_nodes], 0.0, 0.01))

# Hidden layer
# Multiplying input values with weights, before adding biases
# Activating nodes using rectifier (rectified linear unit)
hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, W), b))


# Weights between hidden layer and output layer
# Uniformly chosen in the range [-1.0, 1.0)
w = tf.Variable(tf.random_uniform([number_of_hidden_nodes, 1], -1.0, 1.0))

# Output layer
# Multiplying values from the hidden layers with weights
# Activating layer using hyperbolic tangent function (tanh)
output_layer = tf.tanh(tf.matmul(hidden_layer, w))


# Calculating mean square error (MSE) between excpected output and actual output
mse = tf.reduce_mean(tf.square(tf.sub(y, output_layer)))

# Training network minimizing MSE
train_step = tf.train.GradientDescentOptimizer(0.25).minimize(mse)


# Setting up Tensorflow and initializing variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())


# Running max 5000 iterations
# If no solution is found after 5000 iterations,
# the network will probably oscillate forever
for i in range(5000):
    # Calculate error
    error, _ = sess.run([mse, train_step])

    # If error is sufficiently low, exit loop
    if error < 0.01:
        break
else:
	# Print warning if no solution found in 5000 steps
    print('No solution found')

# Print number of steps, final MSE, and final solution
print('Steps %d: MSE: %f' % (i, error))
print(sess.run(output_layer))
