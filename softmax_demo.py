'''Train a network using the MNIST dataset
'''

import Softmax as net
import data_loaders

def main():

  digits = data_loaders.load_csv('/Users/JordanDodson/code/datasets/mnist/mnist_digits.csv', lambda x: x/255)

  # Initialize a network
  n = net.Network([784, 30, 10])

  # Train the network
  n.sgd(digits[:50000], 10, 30, 0.01, 0, 0, True, digits[50000:60000])

  
if __name__ == '__main__':
  main()
