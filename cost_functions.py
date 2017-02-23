import numpy as np

def cross_entropy(t, a):
  '''
  Return the cross-entropy cost for the batch.

  Each column corresponds to one training instance.

  t: numpy.ndarray --> targets
  a: numpy.ndarray --> activations
  '''

  return -np.sum(t * np.log2(a))


# TODO: Need to use nan_to_num to ensure stability
def binary_entropy(a, t):
  '''Compute the cross entropy given an output activation and target.'''
  return (t - 1).dot(np.log2(a)) - t.dot(np.log2(1-a))
