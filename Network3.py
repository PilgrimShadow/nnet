import numpy as np
import time
from activations import sigmoid, sigmoid_prime

class Network3(object):
  '''Handles batches all at once.
  '''

  def __init__(self, sizes):
    self.num_layers = len(sizes)

    # The sizes of the layers
    self.sizes = sizes

    # Lifetime counter of the number of updates with the training data
    self.num_updates = 0

    # The bias vectors and weight matrices for each layer
    self.biases =  [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # The nabla_b and nabla_w values from the last update
    self.last_nabla_b = [np.zeros((y, 1)) for y in sizes[1:]]
    self.last_nabla_w = [np.zeros((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]


  def cost_derivative(self, a, y):
    return (a - y)

  def cross_entropy_cost_derivative(self, a, y):
    return 

  def feedForward(self, a):
    '''Compute the output of the network given an input vector.
    '''

    for b, w in zip(self.biases, self.weights):
      a = sigmoid(w.dot(a) + b)

    return a


  def sgd(self, training_data, epochs, batch_size, eta, alpha, test_data = None):
    '''Train the network with stochastic gradient descent'''

    # Check for test_data
    if test_data != None:
      show_progress = True
      num_tests = len(test_data)
    else:
      show_progress = False
      num_tests = 0

    # The number of training instances
    n = len(training_data)

    # Update network for each epoch
    for j in range(epochs):

      # Start time for this epoch
      start_time = time.time()

      # Shuffle the training instances
      np.random.shuffle(training_data)

      # Group the training instances by batch size
      batches = (training_data[k : k + batch_size] for k in range(0, n, batch_size))

      # Run through a set of batches
      for batch in batches:

        # Group all input vectors into a single matrix
        xs = np.hstack([instance[0] for instance in batch])

        # Group all target vectors into a single matrix
        ys = np.hstack([instance[1] for instance in batch])

        # Perform a batch update
        self.update_batch((xs, ys), eta, alpha)


      # Compute the elapsed time for this epoch
      epoch_time = time.time() - start_time

      # Increment the update counter
      self.num_updates += 1

      # Test network with test_data
      if show_progress:
        num_correct = 0

        for image, label in test_data:
          activation = self.feedForward(image)
          guess = np.argmax(activation)
          if label[guess] > 0:
            num_correct += 1

        print("Epoch {:d}. {}. {:d} / {:d}".format(j, epoch_time, num_correct, num_tests))
      else:
        # Only print elapsed time if no training data is available
        print("Epoch {:d}. {}.".format(j, epoch_time))


  # batch: 
  # eta: 
  # alpah: 
  #
  def update_batch(self, batch, eta, alpha):

    # The number of instances in this batch
    batch_size = batch[0].shape[1]

    # Update biases and weights for this batch
    for j, nabla_b_j, nabla_w_j in self.backprop(batch):

      self.last_nabla_b[-j] = (-eta * nabla_b_j + alpha * self.last_nabla_b[-j]) / batch_size
      self.last_nabla_w[-j] = (-eta * nabla_w_j + alpha * self.last_nabla_w[-j]) / batch_size

      self.biases[-j]  += self.last_nabla_b[-j]
      self.weights[-j] += self.last_nabla_w[-j]


  def backprop(self, batch):
    '''

      Iterates backwards through layers, yielding nabla_b and nabla_w.

      x : n x 1 ndarray, n is size of input layer
      y : m x 1 ndarray, m is size of output layer
    '''

    # Activations of each layer
    activations = [batch[0]]

    # Weighted inputs for all but the input layer
    weighted_inputs = []

    # forward pass
    for b, w in zip(self.biases, self.weights):

      # broadcasting when adding biases
      weighted_inputs.append(w.dot(activations[-1]) + b)
      activations.append(sigmoid(weighted_inputs[-1]))

    # backward pass initialization
    delta = (self.cost_derivative(activations[-1], batch[1]) * 
                                 sigmoid_prime(weighted_inputs[-1]))

    # Compute the nabla_b and nabla_w terms for the last layer
    nabla_w = sum(np.outer(d, a) for d, a in zip(delta.T, activations[-2].T))
    yield(1, delta.sum(axis=1, keepdims=True), nabla_w)

    # backward pass
    for i in range(2, len(self.sizes)):

      # update delta to be for the previous layer
      delta = (self.weights[-i + 1].T).dot(delta) * sigmoid_prime(weighted_inputs[-i])

      # update nabla_w based on the update delta values
      nabla_w = sum(np.outer(d, a) for d, a in zip(delta.T, activations[-i - 1].T))

      # yield the nabla_b and nabla_w terms for the -i layer
      yield (i, delta.sum(axis=1, keepdims=True), nabla_w)

