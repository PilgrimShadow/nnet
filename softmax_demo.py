'''Train a network using the MNIST dataset
'''

import Softmax as net
from data_loaders import load_csv

def main():

  r = input('Hidden Layers: ')
  layers = [int(x) for x in r.split(' ')]

  r = input('Epochs: ')
  epochs = int(r.strip())

  r = input('Batch size: ')
  batch_size = int(r.strip())

  r = input('Learning rate: ')
  eta = float(r.strip())

  r = input('Momentum: ')
  mu = float(r.strip())

  r = input('Regularization: ')
  lmbda = float(r.strip())

  print('Loading data...')

  digits = load_csv('/Users/JordanDodson/code/datasets/mnist/mnist_digits.csv', lambda x: x/255)

  # Initialize a network
  n = net.Network(digits[:50000], [784] + layers + [10], batch_size, eta, mu, lmbda)

  print('Training...')

  # Train the network
  n.sgd(epochs, stats=True, eval_data=digits[50000:60000])

  while input('Continue training? (y / [n]): ').strip().lower() == 'y':
    r = input('Epochs: ')
    epochs = int(r.strip())
    n.sgd(epochs, stats=True, eval_data=digits[50000:60000])
  
if __name__ == '__main__':
  main()
