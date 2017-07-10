import numpy as np

class ReLU():


  def __init__(self, size, initializer=None):

    self.size = size

    self.biases = np.ndarray(size).reshape(size, 1)

    self.prev = None

    self.initializer = initializer


  def connect(self, prev)

    self.prev = prev

    self.weights = np.ndarray((self.size, self.prev.size))


  def func(self, data):

    


  def deriv(self, data):

    pass


  def backprop(self, deltas):

    pass
