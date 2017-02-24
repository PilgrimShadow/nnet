import struct
import pickle
import numpy as np

# TODO: Extend the open_csv method to handle labels at the beginning and end of a line


def read_images_idx3(file_name):
  with open(file_name, 'rb') as data_file:
    magic_number   = struct.unpack(">i", data_file.read(4))[0]
    image_count    = struct.unpack(">i", data_file.read(4))[0]
    rows_per_image = struct.unpack(">i", data_file.read(4))[0]
    cols_per_image = struct.unpack(">i", data_file.read(4))[0]

    image_size = rows_per_image * cols_per_image
    images = []

    for _ in range(image_count):
      images.append([pixel[0] for pixel in struct.iter_unpack('B', data_file.read(image_size))])

  return images


def read_labels_idx1(file_name):
  data_file = open(file_name, 'rb')

  magic_number = struct.unpack(">i", data_file.read(4))[0]
  label_count  = struct.unpack(">i", data_file.read(4))[0]

  labels = [ label[0] for label in struct.iter_unpack('B', data_file.read()) ]

  data_file.close()

  return labels

def load_pickled_images(filename):
  with open(filename, 'rb') as f:
    images = pickle.load(f)

  for j in range(len(images)):
    images[j] = np.array(images[j]) / 255
    images[j] = images[j].reshape(images[j].size, 1)

  return images


def load_pickled_labels(filename):
  with open(filename, 'rb') as f:
    labels = pickle.load(f)

  for j in range(len(labels)):
    labels[j] = np.concatenate((np.zeros(labels[j]), np.array([1]), np.zeros(9 - labels[j])))
    labels[j] = labels[j].reshape(labels[j].size, 1)

  return labels


def label_to_array(label, n):
  '''Convert a label to an n-element unit vector.'''

  v = np.zeros((n, 1))
  v[label] = 1

  return v


def load_csv(filename, transform):
  '''
  Loads a dataset in csv format.

  Notes
  ~~~~~

  The first column is the target category, a non-negative integer. The remaining
  columns form the input vector, the activations of the input layer.

  Parameters
  ~~~~~~~~~~

  filename: string --> The name of the csv file
  transform: func ---> A transform to apply to the input vectors

  Examples
  ~~~~~~~~
  '''

  dataset = []

  with open(filename) as f:
    for line in f:
      entries  = line.split(',')
      img_data = [float(c) for c in entries[1:]]
      target   = label_to_array(int(entries[0]), 10)
      image    = np.array(list(map(transform, img_data))).reshape((len(img_data), 1))

      dataset.append((image, target))

  return dataset
