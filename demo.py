'''Train a network using the MNIST dataset
'''

import Network3 as net
import data_loaders

def main():

  # Load the MNIST training data
  train_images = data_loaders.load_pickled_images('data/train-images-pickled')
  train_labels = data_loaders.load_pickled_labels('data/train-labels-pickled')

  # Load the MNIST test data
  test_images = data_loaders.load_pickled_images('data/test-images-pickled')
  test_labels = data_loaders.load_pickled_labels('data/test-labels-pickled')

  training_data = list(zip(train_images, train_labels))
  test_data = list(zip(test_images, test_labels))

  # Initialize a network
  n = net.Network3([784, 30, 10])

  # Train the network
  n.sgd(training_data[:50000], 10, 20, 0.05, 0, training_data[50000:])

  
if __name__ == '__main__':
  main()
