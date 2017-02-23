import time
import numpy as np
from activations import relu, relu_prime, softmax
from cost_functions import cross_entropy
from data_loaders import load_csv

# TODO: Add an epoch counter in __init__
# TODO: Add a check for uniformity of layer size

class Network(object):
  '''A feed-forward neural network with ReLU hidden layers and softmax output layer.
  '''

  def __init__(self, train_data, sizes, batch_size, eta, mu, lmbda):
    '''Initialize the members of the network

       sizes: list[int] ------> The sizes of the layers
       train_data: list[(ndarray, ndarray)]------> The training data.
       batch_size: int -------> The size of a mini-batch.
       eta: float ------------> The learning rate.
       mu: float  ------------> The momentum coefficient.
       lmbda: float ----------> The regularization coefficient.
    '''

    # The number of layers
    self.num_layers = len(sizes)

    # The sizes of the layers
    self.sizes = sizes

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

    self.stats = {'epoch_times': [], 'train_costs': [], 'train_errors': [], 'eval_costs': [], 'eval_errors': [],
             'mu': self.mu, 'eta': self.eta, 'lambda': self.lmbda, 'batch_size': self.batch_size, 'train_set_size': len(train_data) }

    # Initialize the bias vectors and weight matrices
    self.biases  = [np.random.uniform(0.05, 0.15, size=(y, 1)) for y in self.sizes[1:]]
    self.weights = [np.random.normal(size=(y, x))/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # Initialize velocities to zero
    self.velocity_b = [np.zeros((y, 1)) for y in self.sizes[1:]]
    self.velocity_w = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]


  def feedForward(self, a):
    '''Compute the output of the network for the given batch.

    Each column is an input vector.

    a : (n x m) ndarray --> initial activations
    '''

    # first layer
    res = self.weights[0].dot(a)
    np.add(res, self.biases[0], res)
    relu(res, res)

    # middle layers
    for b, w in zip(self.biases[1:-1], self.weights[1:-1]):
      if b.shape[0] == res.shape[0]:
        np.dot(w, res, res)
      else:
        res = w.dot(res)

      np.add(res, b, res)
      relu(res, res)

    # Softmax ouput layer
    if self.sizes[-1] == res.shape[0]:
      np.dot(self.weights[-1], res, res)
    else:
      res = self.weights[-1].dot(res)

    np.add(res, self.biases[-1], res)
    softmax(res, res)

    return res


  def error(self, dataset):
    '''Compute the classification error on the given dataset.

    dataset: list[(ndarray, ndarray)]
    '''

    num_correct = 0

    for input, target in dataset:
      output = self.feedForward(input)
      guess = np.argmax(output)
      if target[guess] > 0:
        num_correct += 1

    return 1 - (num_correct / len(dataset))


  def cost(self, dataset):
    '''Cost function

    Compute the cost of the given dataset.

    dataset: list[(ndarray, ndarray)] --> list of training instances
    '''

    base_cost = sum(cross_entropy(t, self.feedForward(x)) for x, t in dataset)
    reg_term  = (self.lmbda / (2 * len(dataset))) * sum(np.sum(np.square(w)) for w in self.weights)

    return base_cost + reg_term


  def sgd(self, epochs, stats = False, eval_data = None):
    '''Train the network with stochastic gradient descent.

    Parameters
    ~~~~~~~~~~

    epochs: int -----------> The number of epochs for which to train.
    stats: bool -----------> Should statistics be computed for each epoch?
    eval_data: list[(ndarray, ndarray)]-------> The evaluation data.

    '''

    # Check for eval_data
    if eval_data is None:
      show_progress = False
      num_tests = 0
    else:
      show_progress = True
      num_tests = len(eval_data)

    # The number of training instances
    n = len(self.train_data)


    # Update network for each epoch
    for j in range(epochs):

      # Start time for this epoch
      start_time = time.time()

      # Shuffle the training instances
      np.random.shuffle(self.train_data)

      # Group the training instances by batch size
      batches = (self.train_data[k : k + self.batch_size] for k in range(0, n, self.batch_size))

      # Run through a set of batches
      for batch in batches:

        # Group all input vectors into a single matrix
        xs = np.hstack([instance[0] for instance in batch])

        # Group all target vectors into a single matrix
        ys = np.hstack([instance[1] for instance in batch])

        # Perform a batch update
        self.update_batch((xs, ys), n)

      # Keep track of epoch times
      self.stats['epoch_times'].append(time.time() - start_time)

      print('Epoch {:{}d} | {:.2f}s'.format(j, int(1 + np.floor(np.log10(epochs))), self.stats['epoch_times'][-1]), end='')

      # Compute the elapsed time for this epoch
      if stats:
        self.stats['train_costs'].append(self.cost(self.train_data))
        self.stats['train_errors'].append(self.error(self.train_data))

        # Test network with eval_data
        if show_progress:

          eval_cost = self.cost(eval_data)
          self.stats['eval_costs'].append(eval_cost)

          eval_error = self.error(eval_data)
          self.stats['eval_errors'].append(eval_error)

          print(" | {:.2f} | {:.2f}%".format(eval_cost, 100*eval_error), end='')

      # Print trailing newline for this epoch
      print('')

    # Return stats for this round of training
    return stats


  def update_batch(self, batch, n):
    '''Update the network with the given mini-batch.

    Parameters
    ~~~~~~~~~~

    batch -> The training instances for the batch
    n -----> The size of the training set.

    '''

    # TODO: Perhaps rename the parameter below, since batch_size is a class member

    # The number of instances in this batch
    batch_size = batch[0].shape[1]

    # Update biases and weights for this batch
    # TODO: the l parameter is kinda ugly
    for l, nabla_b_l, nabla_w_l in self.backprop(batch):

      self.velocity_b[-l] = (self.mu * self.velocity_b[-l] - self.eta * nabla_b_l) / batch_size
      self.velocity_w[-l] = (self.mu * self.velocity_w[-l] - self.eta *
                            (nabla_w_l + (self.lmbda * self.weights[-l]) / n)) / batch_size

      self.biases[-l]  += self.velocity_b[-l]
      self.weights[-l] += self.velocity_w[-l]


  def backprop(self, batch):
    '''Backprop

      Iterates backwards through layers, yielding nabla_b and nabla_w.

      x : n x 1 ndarray, n is size of input layer
      y : m x 1 ndarray, m is size of output layer
    '''

    # Activations of each layer
    activations = [batch[0]]

    # Weighted inputs for all but the input layer
    weighted_inputs = []

    # forward pass
    for b, w in zip(self.biases[:-1], self.weights[:-1]):

      # broadcasting when adding biases
      weighted_inputs.append(w.dot(activations[-1]) + b)
      activations.append(relu(weighted_inputs[-1]))

    # final layer (softmax)
    weighted_inputs.append(self.weights[-1].dot(activations[-1]) + self.biases[-1])
    activations.append(softmax(weighted_inputs[-1]))

    # backward pass initialization
    # employing the cross-entropy cancellation here
    delta = activations[-1] - batch[1]

    # Compute the nabla_b and nabla_w terms for the last layer
    nabla_w = delta.dot(activations[-2].T)
    yield (1, delta.sum(axis=1, keepdims=True), nabla_w)

    # backward pass
    for i in range(2, len(self.sizes)):

      # compute delta for the previous layer
      delta = (self.weights[-i + 1].T).dot(delta) * relu_prime(weighted_inputs[-i])

      # compute nabla_w for the updated delta values
      nabla_w = delta.dot(activations[-(i+1)].T)

      # yield the nabla_b and nabla_w terms for the -i layer
      yield (i, delta.sum(axis=1, keepdims=True), nabla_w)


