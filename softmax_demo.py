'''Train a network using the MNIST dataset
'''

import Softmax as net
from data_loaders import load_csv

def main():

  # Display the available datasets
  print('---Datasets---')
  print('1) MNIST')
  print('2) USPS')
  print('3) Bogazici')

  r = input('Select: ')
  d = int(r.strip())

  print('Loading data...')

  # Load the selected dataset
  if d == 1:
    digits = load_csv('/Users/JordanDodson/code/datasets/mnist/mnist_digits.csv', lambda x: x/255)
    train_data = digits[:50000]
    eval_data  = digits[50000:60000]
    input_layer_size = 784
  elif d == 2:
    digits = load_csv('/Users/JordanDodson/code/datasets/usps_digits/usps_digits.csv', lambda x: (x+1)/2)
    train_data = digits[:7291]
    eval_data  = digits[7291:]
    input_layer_size = 256
  elif d == 3:
    train_data = load_csv('/Users/JordanDodson/code/datasets/bogazici/bogazici-digits-train.csv', lambda x: x/16)
    eval_data  = load_csv('/Users/JordanDodson/code/datasets/bogazici/bogazici-digits-test.csv', lambda x: x/16)
    input_layer_size = 64
    

  # Acquire network params and hyper-params
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


  # Initialize the network
  #n = net.Network(digits[:50000], [784] + layers + [10], batch_size, eta, mu, lmbda, eval_data=digits[50000:60000])
  n = net.Network(train_data, [input_layer_size] + layers + [10], batch_size, eta, mu, lmbda, eval_data=eval_data)

  print('Training...')

  # Train the network
  n.sgd(epochs)

  # Optionally continue training the network
  while input('Continue training? (y / [n]): ').strip().lower() == 'y':
    r = input('Epochs: ')
    epochs = int(r.strip())
    n.sgd(epochs)
  
if __name__ == '__main__':
  main()
