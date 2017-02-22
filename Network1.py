import pickle
import time
import numpy as np
from activations import sigmoid, sigmoid_prime


# Compute the cross entropy given an activation and a target
def cross_entropy(a, y):
  return -(y.dot(np.log2(a)) + (1 - y).dot(np.log2(1-a)))


class Network1(object):

  def __init__(self, sizes):

    # The number of layers in the network
    self.num_layers = len(sizes)

    # The sizes of the layers in the network
    self.sizes = sizes

    # A list of bias vectors (numpy.ndarray)
    self.biases =  [np.random.randn(y, 1) for y in sizes[1:]]

    # A list of weight matrices (numpy.ndarray)
    self.weights = [np.random.randn(numRows, numCols) for numRows, numCols
                                                   in zip(sizes[1:], sizes[:-1])]


  # Evaluate the derivative of the cost function for the
  # given activation and target vectors
  # 
  # a : numpy.ndarray
  # y : numpy.ndarray
  #
  def cost_derivative(self, a, y):
    return a - y


  # Compute the output of the network given an input vector.
  #
  # inputVec : numpy.ndarray
  #
  def feedForward(self, inputVec):

    # Copy the input vector
    a = inputVec.copy()

    for b, w in zip(self.biases, self.weights):
      a = sigmoid(w.dot(a) + b)

    return a


  # Train the network with stochastic gradient descent
  #
  # training_data: 
  # epochs: int
  # batch_size: int
  # eta: float
  # test_data:
  #
  def sgd(self, training_data, epochs, batch_size, eta, test_data = None):

    # Check for test_data
    if test_data != None:
      show_progress = True
      num_tests = len(test_data)
    else:
      show_progress = False
      num_tests = 0

    n = len(training_data)

    # Update network for each epoch
    for j in range(epochs):

      start_time = time.time()

      np.random.shuffle(training_data)

      batches = (training_data[k : k + batch_size] for k in range(0, n, batch_size))

      # Run through a set of batches
      for batch in batches:
        self.update_batch(batch, eta)

      epoch_time = time.time() - start_time

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
        print("Epoch {:d}. {}.".format(j, epoch_time))


  def update_batch(self, batch, eta):

    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    for x, y in batch:
      # Compute gradient for this instance
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)

      # Add result to batch totals
      for j in range(len(nabla_b)):
        nabla_b[j] += delta_nabla_b[j]
        nabla_w[j] += delta_nabla_w[j]

    # Update biases and weights for this batch
    for j in range(len(nabla_b)):
      self.biases[j]  -= (eta / len(batch)) * nabla_b[j]
      self.weights[j] -= (eta / len(batch)) * nabla_w[j]


  # Perform a mini-batch update all at once
  #
  # batch:
  # eta: int
  #
  def update_batch2(self, batch, eta):

    # Compute gradient of cost function for this batch
    nabla_b, nabla_w = self.backprop2(batch)

    # Update biases and weights with gradient
    for j in range(len(nabla_b)):
      self.biases[j]  -= (eta / len(batch)) * nabla_b[j]
      self.weights[j] -= (eta / len(batch)) * nabla_w[j]


  # 
  #
  # x:
  # y:
  #
  def backprop(self, x, y):
    '''

      x : n x 1 ndarray
      y : n x 1 ndarray
    '''

    nabla_b = []
    nabla_w = []

    activations = [x]
    weighted_inputs = []

    # forward pass
    for b, w in zip(self.biases, self.weights):
      weighted_inputs.append(w.dot(activations[-1]) + b)
      activations.append(sigmoid(weighted_inputs[-1]))

    # initialization
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(weighted_inputs[-1])
    nabla_b.append(delta)
    nabla_w.append(np.outer(delta, activations[-2]))

    # backward pass
    for i in range(2, len(self.sizes)):
      delta = (self.weights[-i + 1].T).dot(delta) * sigmoid_prime(weighted_inputs[-i])
      nabla_b.append(delta)
      # check return shape of np.outer
      nabla_w.append(np.outer(delta, activations[-i - 1]))

    # These were built in reverse order
    nabla_w.reverse()
    nabla_b.reverse()

    return (nabla_b, nabla_w)


  # Compute the gradients for all inputs in the batch
  def backprop2(self, batch):

    # Convert to 2d arrays
    xs = []
    ys = []

    for x, y in batch:
      xs.append(x)
      ys.append(y)

    xs = np.array(xs).T
    ys = np.array(ys).T

    # backprop
