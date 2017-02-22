import Network4 as net
import data_loaders
import numpy as np
import json

# The experiment - augmenting training data with noisy images
def main(num_trials, runs_per_trial):

  # load mnist dataset
  digits = data_loaders.load_csv('/Users/JordanDodson/code/datasets/mnist/mnist_digits.csv',
                               lambda x: x/255)

  # a list of lists where we store training results
  stats = []

  # put test images at head of list
  digits.reverse()

  # append random images to tail of list
  for i in range(num_trials*10000):
    digits.append(net.random_training_instance(784, 10))

  # train with various numbers of noisy images
  for i in range(num_trials+1):
    stats.append([])

    # train runs_per_trial networks
    for j in range(runs_per_trial):
      n = net.Network([784, 30, 10])
      s = n.sgd(digits[20000:(70000 + i*10000)], 40, 100, 0.25,
                                                     0.4, 10, True, digits[10000:20000])
      # save the training stats for this run
      stats[-1].append(s)

  # Save results to file
  with open('experiment1_results_{:d}_{:d}.json'.format(num_trials, runs_per_trial), 'w') as f:
    json.dump(stats, f)

  return stats


def process_results(stats):

  a = []

  runs_per_trial = len(stats[0])

  for trial in stats:

    num_epochs = len(trial[0]['epoch_times'])

    a.append({
         'train_errors': np.zeros(num_epochs),
         'eval_errors':  np.zeros(num_epochs)
    })

    for run in trial:
      a[-1]['train_errors'] += np.array(run['train_errors'])
      a[-1]['eval_errors']  += np.array(run['eval_errors'])

    a[-1]['train_errors'] /= runs_per_trial
    a[-1]['eval_errors']  /= runs_per_trial

  return a


if __name__ == '__main__':
  main(6, 3)
