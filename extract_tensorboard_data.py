# adapted from GitHub user tomrunia
# original file: tensorflow_log_loader.py at gist.github.com/tomrunia/1e1d383fb21841e8f144

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt

from sys import argv
from collections import namedtuple


def get_from_commandline_args(count, args_string=None):
    if len(argv) - 1 != count:
        print("You need to enter exactly %i command line arguments%s, "
              "but found %i" % (count, '' if args_string is None else " (%s)" % args_string, len(argv) - 1))
        exit(1)

    return tuple(argv[1:])


# get prefix and model name from command line
def get_prefix_and_model_name():
    return get_from_commandline_args(2, "'prefix' and 'model_name'")


def load_tensorflow_log(path):

    # # Loading too much data is slow...
    # tf_size_guidance = {
    #     'compressedHistograms': 10,
    #     'images': 0,
    #     'scalars': 100,
    #     'histograms': 1
    # }

    # event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    return event_acc


def print_log_contents(event_acc):
    # Show all tags in the log file
    print(event_acc.Tags())


def get_scalars(event_acc):
    Scalar = namedtuple('Scalar', ('name', 'values'))
    return [Scalar(s, np.asarray(event_acc.Scalars(s)).swapaxes(0, 1)[2]) for s in event_acc.Tags()['scalars']]


def get_histograms(event_acc):
    # to be plotted with plt.plot(h.buckets, h.count) ; plt.xlim(-1,1)
    Histogram = namedtuple('Histogram', ('name', 'buckets', 'count'))
    return [Histogram(s, np.asarray(event_acc.Histograms(s))[0][2][5], np.asarray(event_acc.Histograms(s))[0][2][6])
            for s in event_acc.Tags()['histograms']]


def untested_plot(event_acc):
    training_accuracies = event_acc.Scalars('training-accuracy')
    validation_accuracies = event_acc.Scalars('validation_accuracy')

    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2]  # value
        y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:, 0], label='training accuracy')
    plt.plot(x, y[:, 1], label='validation accuracy')

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


if __name__ == '__main__':
    prefix, model_name = get_prefix_and_model_name()

    # path declarations
    checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix
    checkpoint_dir += r"/%s" % model_name
    tensorboard_dir = checkpoint_dir + r"/tensorboard"

    log_file = tensorboard_dir

    print("Loading tensorboard data … (this may take a long time)")
    eva = load_tensorflow_log(log_file)
    print("Done. Log contents:")
    print_log_contents(eva)

    print("Converting data …")
    scalars = get_scalars(eva)
    histograms = get_histograms(eva)

    # store scalars
    for s in scalars:
        npyname = s.name + '_' + model_name
        print("Storing '%s' at %s …" % (s.name, npyname))
        np.save(checkpoint_dir + "/" + npyname, s.values)

    # store histograms
    # TODO

    print("Done, exiting.")
