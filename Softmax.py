# Stdlib
import time
import numpy as np

# Project
from activations import relu, relu_prime, softmax
from cost_functions import cross_entropy
from data_loaders import load_csv

# TODO: Rewrite _backprop() to use _backprop_activations
# TODO: Instead of a list of weight matrices, try one single weight tensor
# TODO: Optimize memory usage. Reuse ndarrays wherever possible
# TODO: Add a check for uniformity of layer size

class Network(object):
  '''A feed-forward neural network with ReLU hidden layers and softmax output layer.
  '''

  def __init__(self, train_data, hidden_sizes, batch_size, eta, mu, lmbda, eval_data=None, keep_stats=True):
    '''Initialize the members of the network

       hidden_sizes: list[int] ------> The sizes of the hidden layers
       train_data: list[(ndarray, ndarray)]------> The training data (input, target)
       batch_size: int -------> The size of a mini-batch.
       eta: float ------------> The learning rate.
       mu: float  ------------> The momentum coefficient.
       lmbda: float ----------> The regularization coefficient.
       eval_data: list[(ndarray, ndarray)]-------> The evaluation data.
       keep_stats: bool ------> Should statistics be computed for each epoch?
    '''

    # The sizes of the layers
    self.sizes = [len(train_data[0][0])] + hidden_sizes + [len(train_data[0][1])]

    # The number of layers
    self.num_layers = len(self.sizes)

    # The data used to train the network
    self.train_data = train_data

    # The size of an SGD mini-batch
    self.batch_size = batch_size

    # The learning rate
    self.eta = eta

    # The momentum coefficient
    self.mu = mu

    # The regularization parameter
    self.lmbda = lmbda

    # The number of epochs this network has been trained
    self.epochs_trained = 0

    # The evaluation data
    self.eval_data = eval_data

    # Boolean indicating whether stats should be kept during training
    self.keep_stats = keep_stats

    # Memory to store layer activations
    self._activations = [np.ndarray(shape=(y, 1)) for y in self.sizes[1:]]

    # Memory to store batch activations
    self._batch_activations = [np.ndarray(shape=(y, self.batch_size)) for y in self.sizes[1:]]

    # Memory to store weighted inputs for each layer
    self._weighted_inputs = [np.ndarray(shape=(y, self.batch_size)) for y in self.sizes[1:]]

    # Dictionary to track the training stats
    self.stats = {'epoch_times': [], 'train_costs': [], 'train_errors': [], 'eval_costs': [], 'eval_errors': [],
             'mu': self.mu, 'eta': self.eta, 'lambda': self.lmbda, 'batch_size': self.batch_size, 'train_set_size': len(train_data)}

    # Initialize the bias vectors and weight matrices
    self.biases  = [np.random.uniform(0.05, 0.15, size=(y, 1)) for y in self.sizes[1:]]
    self.weights = [np.random.normal(size=(y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # Scale the initial weights
    for w in self.weights:
      np.divide(w, np.sqrt(w.shape[1]), out=w)

    # Initialize velocities to zero
    self.velocity_b = [np.zeros((y, 1)) for y in self.sizes[1:]]
    self.velocity_w = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]


  def feedForward(self, a):
    '''Compute the output of the network for the given batch.

    Each column is an input vector.

    a : (n x 1) ndarray --> initial activation
    '''

    # first layer
    np.dot(self.weights[0], a, out=self._activations[0])
    np.add(self._activations[0], self.biases[0], self._activations[0])
    relu(self._activations[0], out=self._activations[0])

    # middle layers
    for i in range(1, len(self.weights)-1):
      np.dot(self.weights[i], self._activations[i-1], self._activations[i])
      np.add(self._activations[i], self.biases[i], self._activations[i])
      relu(self._activations[i], self._activations[i])

    # Softmax ouput layer
    np.dot(self.weights[-1], self._activations[-2], self._activations[-1])
    np.add(self._activations[-1], self.biases[-1], self._activations[-1])
    softmax(self._activations[-1], self._activations[-1])

    return self._activations[-1]


  def cost_and_error(self, dataset):

    num_correct = 0
    base_cost = 0

    for x, t in dataset:
      output = self.feedForward(x)
      base_cost += cross_entropy(t, output)
      if t[np.argmax(output)] > 0:
        num_correct += 1

    reg_term  = (self.lmbda / (2 * len(dataset))) * sum(np.sum(np.square(w)) for w in self.weights)

    return (base_cost + reg_term, 1 - num_correct / len(dataset))


  def error(self, dataset):
    '''Compute the classification error on the given dataset.

    dataset: list[(ndarray, ndarray)] --> (input, target)
    '''

    num_correct = sum( t[np.argmax(self.feedForward(x))] > 0 for x, t in dataset )

    return 1 - (num_correct / len(dataset))


  def cost(self, dataset):
    '''Cost function

    Compute the cost of the given dataset.

    dataset: list[(ndarray, ndarray)] --> list of training instances (input, target)
    '''

    base_cost = sum(cross_entropy(t, self.feedForward(x)) for x, t in dataset)
    reg_term  = (self.lmbda / (2 * len(dataset))) * sum(np.sum(np.square(w)) for w in self.weights)

    return base_cost + reg_term


  def sgd(self, epochs):
    '''Train the network with stochastic gradient descent.

    Parameters
    ~~~~~~~~~~

    epochs: int -----------> The number of epochs for which to train.

    '''

    # Check for eval_data
    if self.eval_data is None:
      show_progress = False
      num_tests = 0
    else:
      show_progress = True
      num_tests = len(self.eval_data)

    # Used for formatting output
    f = self.epochs_trained + epochs

    # Update network for each epoch
    for j in range(epochs):

      # Start time for this epoch
      start_time = time.time()

      # Shuffle the training instances
      np.random.shuffle(self.train_data)

      # Group the training instances by batch size
      batches = (self.train_data[k : k + self.batch_size] for k in range(0, len(self.train_data), self.batch_size))

      # Run through a set of batches
      for batch in batches:

        # Group all input vectors into a single matrix
        xs = np.hstack([instance[0] for instance in batch])

        # Group all target vectors into a single matrix
        ys = np.hstack([instance[1] for instance in batch])

        # Ignoring any remainder batch for now
        if xs.shape[1] == self.batch_size:
          # Perform a batch update
          self._update_batch((xs, ys))

      # Another epoch has been completed
      self.epochs_trained += 1

      # Keep track of epoch times
      self.stats['epoch_times'].append(time.time() - start_time)

      # Display the elapsed time for this epoch
      print('Epoch {:{}d} | {:.4f}s'.format(self.epochs_trained, int(1 + np.floor(np.log10(f))), self.stats['epoch_times'][-1]), end='')

      # Compute the elapsed time for this epoch
      if self.keep_stats:

        train_cost, train_error = self.cost_and_error(self.train_data)
        self.stats['train_costs'].append(train_cost)
        self.stats['train_errors'].append(train_error)

        # Test network with eval_data
        if show_progress:

          eval_cost, eval_error = self.cost_and_error(self.eval_data)
          self.stats['eval_costs'].append(eval_cost)
          self.stats['eval_errors'].append(eval_error)

          print(" | {:.2f} | {:.2f}%".format(eval_cost, 100*eval_error), end='')

      # Print trailing newline for this epoch
      print('')

    # Return the cumulative training stats for the network
    return self.stats


  def _update_batch(self, batch):
    '''Update the network with the given mini-batch.

    Parameters
    ~~~~~~~~~~

    batch -> The training instances for the batch
    n -----> The size of the training set.

    '''

    # TODO: Perhaps rename the parameter below, since batch_size is a class member

    # The number of instances in this batch
    batch_size = batch[0].shape[1]

    n = len(self.train_data)

    # Factors used in the update equations
    veloc_factor = self.mu / batch_size
    decay_factor = self.eta * self.lmbda / (n * batch_size)
    grad_factor = self.eta / batch_size

    # Update biases and weights for this batch
    # TODO: the l parameter is kinda ugly
    for l, nabla_b_l, nabla_w_l in self._backprop(batch):

      # Update the bias velocity
      np.multiply(self.velocity_b[-l], veloc_factor, out=self.velocity_b[-l])
      np.multiply(nabla_b_l, grad_factor, out=nabla_b_l)
      np.subtract(self.velocity_b[-l], nabla_b_l, out=self.velocity_b[-l])

      # Update the weight velocity
      np.multiply(self.velocity_w[-l], veloc_factor, out=self.velocity_w[-l])
      np.multiply(nabla_w_l, grad_factor, out=nabla_w_l)
      np.subtract(self.velocity_w[-l], nabla_w_l, out=self.velocity_w[-l])

      # Update the biases
      np.add(self.biases[-l], self.velocity_b[-l], out=self.biases[-l])

      # Update the weights
      np.multiply(self.weights[-l], 1 - decay_factor, out=self.weights[-l])
      np.add(self.weights[-l], self.velocity_w[-l], out=self.weights[-l])


  def _backprop(self, batch,):
    '''Backprop

      Iterates backwards through layers, yielding nabla_b and nabla_w.

      x : n x 1 ndarray, n is size of input layer
      y : m x 1 ndarray, m is size of output layer
    '''

    # first layer
    np.dot(self.weights[0], batch[0], out=self._weighted_inputs[0])
    np.add(self._weighted_inputs[0], self.biases[0], out=self._weighted_inputs[0])
    relu(self._weighted_inputs[0], out=self._batch_activations[0])

    # hidden layers
    for i in range(1, len(self.weights)-1):
      np.dot(self.weights[i], self._batch_activations[i-1], out=self._weighted_inputs[i])
      np.add(self._weighted_inputs[i], self.biases[i], out=self._weighted_inputs[i])
      relu(self._weighted_inputs[i], out=self._batch_activations[i])

    # final layer (softmax)
    np.dot(self.weights[-1], self._batch_activations[-2], out=self._weighted_inputs[-1])
    np.add(self._weighted_inputs[-1], self.biases[-1], out=self._weighted_inputs[-1])
    softmax(self._weighted_inputs[-1], out=self._batch_activations[-1])

    # TODO: Create delta and nabla_w tensors to reuse in all calculations below

    # backward pass initialization
    # We store delta[i] in self._batch_activations[i] to reduce memory_usage
    np.subtract(self._batch_activations[-1], batch[1], out=self._batch_activations[-1])

    # Compute the nabla_b and nabla_w terms for the last layer
    nabla_w = self._batch_activations[-1].dot(self._batch_activations[-2].T)
    yield (1, self._batch_activations[-1].sum(axis=1, keepdims=True), nabla_w)

    # backward pass
    for i in range(2, len(self.sizes)-1):

      # compute delta for the previous layer
      np.dot(self.weights[-i+1].T, self._batch_activations[-i+1], out=self._batch_activations[-i])
      relu_prime(self._weighted_inputs[-i], out=self._weighted_inputs[-i])
      np.multiply(self._batch_activations[-i], self._weighted_inputs[-i], out=self._batch_activations[-i])

      # compute nabla_w for the updated delta values
      nabla_w = self._batch_activations[-i].dot(self._batch_activations[-i-1].T)

      # yield the nabla_b and nabla_w terms for the -i layer
      yield (i, self._batch_activations[-i].sum(axis=1, keepdims=True), nabla_w)

    np.dot(self.weights[-len(self.sizes)+2].T, self._batch_activations[-len(self.sizes)+2], out=self._batch_activations[-len(self.sizes)+1])
    relu_prime(self._weighted_inputs[-len(self.sizes)+1], out=self._weighted_inputs[-len(self.sizes)+1])
    np.multiply(self._batch_activations[-len(self.sizes)+1], self._weighted_inputs[-len(self.sizes)+1], out=self._batch_activations[-len(self.sizes)+1])
    
    nabla_w = self._batch_activations[-len(self.sizes)+1].dot(batch[0].T)
    yield (len(self.sizes)-1, self._batch_activations[-len(self.sizes)+1].sum(axis=1, keepdims=True), nabla_w)



