import numpy as np

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
  '''The softmax function.

  Each column is an activation vector.

  z   : numpy.ndarray --> array to which the function is applied
  out : numpy.ndarray --> where to store the result
  '''

  # We subtract a column's max for numerical stability
  m = np.amax(z, axis=0)

  if out is None:
    e = np.exp(z-m)
    return e / np.sum(e, axis=0)
  else:
    np.subtract(z, m, out)
    np.exp(z, out)
    np.divide(out, np.sum(out, axis=0), out)
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


