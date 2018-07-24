# adapted from GitHub user tomrunia
# original file: tensorflow_log_loader.py at gist.github.com/tomrunia/1e1d383fb21841e8f144

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt

from sys import argv
from collections import namedtuple
import pickle as pd


# define named tuples globally for pickle
Scalar = namedtuple('Scalar', ('name', 'values'))
Histogram = namedtuple('Histogram', ('name', 'steps'))
HistogramStep = namedtuple('HistogramStep', ('name', 'step', 'bucket_low_limit', 'bucket_filling'))
CompressedHistogram = namedtuple('CompressedHistogram', ('name', 'steps',
                                                         'Infm', 'stdm3', 'stdm2', 'stdm1',
                                                         'median',
                                                         'stdp1', 'stdp2', 'stdp3', 'Infp'))


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

    tf_size_guidance = {
        'distributions': 10000,
        'scalars': 10000,
        'histograms': 10000
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    # event_acc = EventAccumulator(path)
    event_acc.Reload()
    return event_acc


def print_log_contents(event_acc):
    # Show all tags in the log file
    print(event_acc.Tags())


def get_scalars(event_acc):
    return [Scalar(s, np.asarray(event_acc.Scalars(s)).swapaxes(0, 1)[2].tolist()) for s in event_acc.Tags()['scalars']]


def get_histograms(event_acc):
    # to be plotted with plt.plot(h.bucket_low_limit, h.bucket_filling) ; plt.xlim(-1,1) // how handle low limit?

    hist_list = []
    for h in event_acc.Tags()['histograms']:
        step_list = [HistogramStep(h, hist_step.step, hist_step[2][5], hist_step[2][6])
                     for hist_step in event_acc.Histograms(h)]

        hist_list.append(Histogram(h, step_list))

    return hist_list


def get_compressed_histograms(event_acc):
    # Normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
    # naturally gives bands around median of width 1 std dev, 2 std dev, 3 std dev,
    # and then the long tail.
    # NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)


    chist_list = []
    for ch_name in event_acc.Tags()['distributions']:
        name = ch_name
        ch = event_acc.CompressedHistograms(ch_name)
        steps = []
        Infm = []
        stdm3 = []
        stdm2 = []
        stdm1 = []
        median = []
        stdp1 = []
        stdp2 = []
        stdp3 = []
        Infp = []
        for ch_step in ch:
            steps.append(ch_step.step)
            Infm.append(ch_step.compressed_histogram_values[0][1])
            stdm3.append(ch_step.compressed_histogram_values[1][1])
            stdm2.append(ch_step.compressed_histogram_values[2][1])
            stdm1.append(ch_step.compressed_histogram_values[3][1])
            median.append(ch_step.compressed_histogram_values[4][1])
            stdp1.append(ch_step.compressed_histogram_values[5][1])
            stdp2.append(ch_step.compressed_histogram_values[6][1])
            stdp3.append(ch_step.compressed_histogram_values[7][1])
            Infp.append(ch_step.compressed_histogram_values[8][1])
        chist_list.append(CompressedHistogram(name, steps,
                                              Infm, stdm3, stdm2, stdm1,
                                              median,
                                              stdp1, stdp2, stdp3, Infp))

    return chist_list


def save_to_dir(directory, scalars, histograms, compressed_histograms, prefix=None):
    if not directory.endswith("/"):
        directory += "/"
    if prefix is None:
        prefix = ""
    else:
        prefix = "_" + prefix
    names = []

    # store scalars
    for s in scalars:
        npyname = model_name + prefix + "_scalar_" + s.name + ".pkl"
        print("Storing scalar '%s' at %s …" % (s.name, npyname))
        with open(directory + npyname, 'wb') as f:
            pd.dump(s, f)
        names.append(directory + npyname)

    for h in histograms:
        pklname = model_name + prefix + "_hist_" + h.name + ".pkl"
        print("Storing histogram '%s' at %s …" % (h.name, pklname))
        with open(directory + pklname, 'wb') as f:
            pd.dump(h, f)
        names.append(directory + pklname)

    for ch in compressed_histograms:
        pklname = model_name + prefix + "_comp_hist_" + ch.name + ".pkl"
        print("Storing compressed histogram '%s' at %s …" % (ch.name, pklname))
        with open(directory + pklname, 'wb') as f:
            pd.dump(ch, f)
        names.append(directory + pklname)

    return names


if __name__ == '__main__':
    prefix, model_name = get_prefix_and_model_name()

    # path declarations
    checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix
    checkpoint_dir += r"/%s" % model_name
    tensorboard_dir = checkpoint_dir + r"/tensorboard"
    validation_dir = tensorboard_dir + r"/validation"

    print("\nLoading training tensorboard data … (this may take a long time)")
    eva = load_tensorflow_log(tensorboard_dir)
    print("Done. Log contents:")
    print_log_contents(eva)

    print("\nConverting data …")
    scalars = get_scalars(eva)
    histograms = get_histograms(eva)
    compressed_histograms = get_compressed_histograms(eva)

    path_list = save_to_dir(tensorboard_dir, scalars, histograms, compressed_histograms)

    print("\nLoading validation tensorboard data … (this may take a long time, but shorter than for training)")
    eva = load_tensorflow_log(validation_dir)
    print("Done. Log contents:")
    print_log_contents(eva)

    print("\nConverting data …")
    scalars = get_scalars(eva)
    histograms = get_histograms(eva)
    compressed_histograms = get_compressed_histograms(eva)

    path_list.extend(save_to_dir(validation_dir, scalars, histograms, compressed_histograms, prefix='validation'))

    print("Done.")
    print("\nThe following files have been created:")
    for name in path_list:
        print(name)
    print()
