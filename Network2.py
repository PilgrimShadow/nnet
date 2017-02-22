import numpy as np
from activations import sigmoid, sigmoid_prime

class Network2(object):
  '''
  '''

  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases =  [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


  def cost_derivative(self, a, y):
    return (a - y)


  def feedForward(self, a):
    '''Compute the output of the network given an input vector.
    '''

    for b, w in zip(self.biases, self.weights):
      a = sigmoid(w.dot(a) + b)

    return a


  def sgd(self, training_data, epochs, batch_size, eta, test_data = None):
    '''Train the network with stochastic gradient descent'''

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

      # Start time for this epoch
      start_time = time.time()

      np.random.shuffle(training_data)

      batches = (training_data[k : k + batch_size] for k in range(0, n, batch_size))

      # Run through a set of batches
      for batch in batches:
        self.update_batch(batch, eta)

      # Elapsed time for this epoch
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
    '''
    '''

    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    for x, y in batch:

      # Iterate backwards through layers
      for j, delta_nabla_b_j, delta_nabla_w_j in self.backprop(x, y):
        nabla_b[-j] += delta_nabla_b_j
        nabla_w[-j] += delta_nabla_w_j

    # Update biases and weights for this batch
    for j in range(len(nabla_b)):
      self.biases[j]  -= (eta / len(batch)) * nabla_b[j]
      self.weights[j] -= (eta / len(batch)) * nabla_w[j]


  def backprop(self, x, y):
    '''

      Iterates backwards through layers, yielding nabla_b and nabla_w.

      x : n x 1 ndarray, n is size of input layer
      y : m x 1 ndarray, m is size of output layer
    '''

    activations = [x]
    weighted_inputs = []

    # forward pass
    for b, w in zip(self.biases, self.weights):
      weighted_inputs.append(w.dot(activations[-1]) + b)
      activations.append(sigmoid(weighted_inputs[-1]))

    # backward pass initialization
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(weighted_inputs[-1])
    yield(1, delta, np.outer(delta, activations[-2]))

    # backward pass
    for i in range(2, len(self.sizes)):
      delta = (self.weights[-i + 1].T).dot(delta) * sigmoid_prime(weighted_inputs[-i])
      yield (i, delta, np.outer(delta, activations[-i - 1]))

