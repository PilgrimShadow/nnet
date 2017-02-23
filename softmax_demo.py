'''Train a network using the MNIST dataset
'''

import Softmax as net
from data_loaders import load_csv

def main():

  digits = load_csv('/Users/JordanDodson/code/datasets/mnist/mnist_digits.csv', lambda x: x/255)

  # Initialize a network
  n = net.Network([784, 30, 10])

  # Train the network
  n.sgd(digits[:50000], 15, 30, eta=0.01, mu=0, lmbda=10, stats=True, eval_data=digits[50000:60000])

  
if __name__ == '__main__':
  main()
