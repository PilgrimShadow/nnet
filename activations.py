import numpy as np

class ActivationFunction():

  @staticmethod
  def func(z, out=None):
    pass

  @staticmethod
  def prime(z, out=None):
    pass


class Sigmoid(ActivationFunction):

  def func(z, out=None):
    if out is None:
      return 1 / (1 + np.exp(-z))
    else:
      # perform all operations in-place
      np.negative(z, out)
      np.exp(out, out)
      np.add(out, 1, out)
      np.reciprocal(out, out)
      return out

  def prime(z, out=None):
    if out is None:
      return sigmoid(z) * (1 - sigmoid(z))
    else:
      tmp1 = sigmoid(z)
      np.subtract(1, tmp1, out)
      np.multiply(tmp1, out, out)
      return out


class Relu(ActivationFunction):

  def func(z, out=None):
    if out is None:
      return np.maximum(0, z) 
    else:
      return np.maximum(0, z, out)

  def prime(z, out=None):
    if out is None:
      return np.greater_equal(z, 0)
    else:
      return np.greater_equal(z, 0, out)


class Elu(ActivationFunction):

  def func(z, out=None):

    # Where to store the result
    a = z.copy() if out is None else out

    # The positive part
    np.maximum(0, z, a)

    # The negative part
    b = np.minimum(0, z)
    np.exp(b, b)
    np.subtract(b, 1)

    # Combine
    np.add(a, b, a)

    return a

  def prime(z, out=None):

    # Where to store the result
    a = z.copy() if out is None else out

    # The positive part
    np.greater(z, 0, a)

    # The negative part
    b = np.minimum(0, z)
    np.exp(b, b)

    # Combine
    np.add(a, b, a)

    return a


class Tanh(ActivationFunction):

  def func(z, out=None):

    # Where to store the result
    a = z.copy() if out is None else out

    # Scale the input to prevent overflow
    m = np.amax(z, axis=0)
    np.subtract(z, m, a)

    # Compute Tanh
    np.multiply(a, 2, a)
    np.exp(a, a)
    np.subtract(a, 1, a)
    np.divide(a, a+2, a)

    return a


  def prime(z, out=None):

    # Where to store the result
    a = z.copy() if out is None else out

    Tanh.func(z, a)
    np.multiply(a, a, a)
    np.subtract(1, a, a)

    return a


def sigmoid(z, out=None):
  '''Element-wise logistic sigmoid.

     z   : ndarray --> array to which the function is applied
     out : ndarray --> where to store result
  '''

  if out is None:
    return 1 / (1 + np.exp(-z))
  else:
    # perform all operations in-place
    np.negative(z, out)
    np.exp(out, out)
    np.add(out, 1, out)
    np.reciprocal(out, out)
    return out


# Must be careful to avoid overflow
def sigmoid_prime(z, out=None):
  '''Element-wise gradient of logistic sigmoid.

     z   : ndarray --> array to which the function is applied
     out : ndarray --> where to store result
  '''
  if out is None:
    return sigmoid(z) * (1 - sigmoid(z))
  else:
    tmp1 = sigmoid(z)
    np.subtract(1, tmp1, out)
    np.multiply(tmp1, out, out)
    return out


def softmax(z, out=None):
  '''The softmax function

  Each column is an activation vector.

  z   : numpy.ndarray --> array or mini-batch to which the function is applied
  out : numpy.ndarray --> where to store the result
  '''

  # We subtract a column's max for numerical stability
  m = np.amax(z, axis=0)

  if out is None:
    e = np.exp(z-m)
    np.divide(e, np.sum(e, axis=0), out=e)
    return e
  else:
    np.subtract(z, m, out)
    np.exp(out, out)
    np.divide(out, np.sum(out, axis=0), out=out)
    return out


def softmax_prime(z, out=None):
  '''The derivative of the softmax function.

  Each column is an activation vector.

  z   : numpy.ndarray --> array to which the function is applied
  out : numpy.ndarray --> where to store the result
  '''

  pass

def relu(z, out=None):
  '''Rectified linear unit.

  Each column is an activation vector.

  z   : ndarray --> array to which the function is applied
  out : ndarray --> where to store the result
  '''

  if out is None:
    return np.maximum(0, z) 
  else:
    return np.maximum(0, z, out)


def relu_prime(z, out=None):
  '''
  Return the gradient of the ReLU activation function.

  z   : numpy.ndarray --> array to which the function is applied
  out : numpy.ndarray --> where to store the result
  '''

  if out is None:
    return np.greater_equal(z, 0)
  else:
    return np.greater_equal(z, 0, out)


