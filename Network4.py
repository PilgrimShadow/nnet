import time
import numpy as np
from activations import *
from data_loaders import load_csv


class Network(object):
  '''An feed-forward neural network with sigmoid neurons.'''

  def __init__(self, sizes):
    self.num_layers = len(sizes)

    # The sizes of the layers
    self.sizes = sizes

    # Initialize weights and biases
    self.initialize()


  def initialize(self):
    '''Initialize network weights and biases, avoiding saturation.'''

    # The bias vectors and weight matrices for each layer
    self.biases  = [np.random.normal(size=(y, 1)) for y in self.sizes[1:]]
    self.weights = [np.random.normal(size=(y, x))/np.sqrt(x)
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # Initialize velocities to zero
    self.velocity_b = [np.zeros((y, 1)) for y in self.sizes[1:]]
    self.velocity_w = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]


  def initialize_large(self):
    '''Naively initialize network weights and biases.'''

    # The bias vectors and weight matrices for each layer
    self.biases  = [np.random.normal(size=(y, 1)) for y in self.sizes[1:]]
    self.weights = [np.random.normal(size=(y, x))
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # The nabla_b and nabla_w values from the last update
    self.velocity_b = [np.zeros((y, 1)) for y in self.sizes[1:]]
    self.velocity_w = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]


  def feedForward(self, a):
    '''Compute the output of the network given an input vector (or batch matrix).

       A batch matrix is a numpy 2d-array with each column an input vector.
    '''

    for b, w in zip(self.biases, self.weights):
      tmp1 = w.dot(a)
      np.add(tmp1, b, tmp1)

      if a.shape == tmp1.shape:
        sigmoid(tmp1, a)
      else:
        a = sigmoid(tmp1)

    return a


  def error(self, dataset):
    '''Compute the classification error on the given dataset.

       A dataset is a list of (input, target) pairs.
    '''

    num_correct = 0

    for input, target in dataset:
      output = self.feedForward(input)
      guess = np.argmax(output)
      if target[guess] > 0:
        num_correct += 1

    return 1 - (num_correct / len(dataset))


  def cost(self, dataset):
    '''Compute the cost of the given dataset.'''

    cost = 0

    for x, y in dataset:
      a = self.feedForward(x)
      cost -= (y.T.dot(np.log(a))[0][0] + (1-y).T.dot(np.log(1-a))[0][0])

    return cost


  def sgd(self, train_data, epochs, batch_size, eta, mu, lmbda,
                stats = False, eval_data = None):
    '''Train the network with stochastic gradient descent.

    Parameters
    ~~~~~~~~~~

    train_data ------> The training data.
    epochs ----------> The number of epochs for which to train.
    batch_size ------> The size of a mini-batch.
    eta -------------> The learning rate.
    mu --------------> The momentum coefficient.
    lmbda -----------> The regularization coefficient.
    stats -----------> Should statistics be computed for each epoch?
    eval_data -------> The evaluation data.

    '''

    # Check for eval_data
    if eval_data != None:
      show_progress = True
      num_tests = len(eval_data)
    else:
      show_progress = False
      num_tests = 0

    epoch_times  = []
    train_costs  = []
    train_errors = []
    eval_costs   = []
    eval_errors  = []

    # The number of training instances
    n = len(train_data)

    # Update network for each epoch
    for j in range(epochs):

      # Start time for this epoch
      start_time = time.time()

      # Shuffle the training instances
      np.random.shuffle(train_data)

      # Group the training instances by batch size
      batches = (train_data[k : k + batch_size] for k in range(0, n, batch_size))

      # Run through a set of batches
      for batch in batches:

        # Group all input vectors into a single matrix
        xs = np.hstack([instance[0] for instance in batch])

        # Group all target vectors into a single matrix
        ys = np.hstack([instance[1] for instance in batch])

        # Perform a batch update
        self.update_batch((xs, ys), eta, mu, lmbda, n)

      # Keep track of epoch times
      epoch_times.append(time.time() - start_time)

      # Compute the elapsed time for this epoch
      if stats:
        train_costs.append(self.cost(train_data))

        train_errors.append(self.error(train_data))

        # Test network with eval_data
        if show_progress:

          eval_cost = self.cost(eval_data)
          eval_costs.append(eval_cost)

          eval_error = self.error(eval_data)
          eval_errors.append(eval_error)

          print("Epoch {:d} | {:.2f}s | {:.2f} | {:.2f}%".format(j,
                epoch_times[-1], eval_cost, 100*eval_error))
        else:
          print("Epoch {:d} | {:.2f}s".format(j, epoch_times[-1]))

    # Return stats for this round of training
    return { "epoch_times": epoch_times, "train_costs": train_costs,
             "train_errors": train_errors, "eval_costs": eval_costs,
             "eval_errors": eval_errors, "mu": mu, "eta": eta,
             "lambda": lmbda, "batch_size": batch_size, "train_set_size": n }


  def update_batch(self, batch, eta, mu, lmbda, n):
    '''Update the network with the given mini-batch.

    Parameters
    ~~~~~~~~~~

    batch -> The training instances for the batch
    eta ---> The learning rate.
    mu ----> The momentum coefficient.
    lmbda -> The regularization parameter.
    n -----> The size of the training set.

    '''

    # The number of instances in this batch
    batch_size = batch[0].shape[1]

    # Update biases and weights for this batch
    for l, nabla_b_l, nabla_w_l in self.backprop(batch):

      self.velocity_b[-l] = (mu * self.velocity_b[-l] - eta * nabla_b_l) / batch_size
      self.velocity_w[-l] = (mu * self.velocity_w[-l] - eta *
                            (nabla_w_l + (lmbda * self.weights[-l]) / n)) / batch_size

      self.biases[-l]  += self.velocity_b[-l]
      self.weights[-l] += self.velocity_w[-l]


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
    # employing the cross-entropy cancellation here
    delta = activations[-1] - batch[1]

    # Compute the nabla_b and nabla_w terms for the last layer
    nabla_w = delta.dot(activations[-2].T)
    yield(1, delta.sum(axis=1, keepdims=True), nabla_w)

    # backward pass
    for i in range(2, len(self.sizes)):

      # update delta to be for the previous layer
      delta = (self.weights[-i + 1].T).dot(delta) * sigmoid_prime(
                                       weighted_inputs[-i], weighted_inputs[-i])

      # update nabla_w based on the update delta values
      nabla_w = delta.dot(activations[-(i+1)].T)

      # yield the nabla_b and nabla_w terms for the -i layer
      yield (i, delta.sum(axis=1, keepdims=True), nabla_w)



