import numpy as np

import matplotlib.pyplot as plt
import plot_helper

from sys import argv
from collections import namedtuple
import pickle as pd

from extract_tensorboard_data import Histogram, HistogramStep, Scalar, CompressedHistogram


# # define named tuples globally for pickle
# Scalar = namedtuple('Scalar', ('name', 'values'))
# Histogram = namedtuple('Histogram', ('name', 'steps'))
# HistogramStep = namedtuple('HistogramStep', ('name', 'step', 'bucket_low_limit', 'bucket_filling'))
# CompressedHistogram = namedtuple('CompressedHistogram', ('name', 'steps',
#                                                          'Infm', 'stdm3', 'stdm2', 'stdm1',
#                                                          'median',
#                                                          'stdp1', 'stdp2', 'stdp3', 'Infp'))


def get_from_commandline_args(count, args_string=None):
    if len(argv) - 1 != count:
        print("You need to enter exactly %i command line arguments%s, "
              "but found %i" % (count, '' if args_string is None else " (%s)" % args_string, len(argv) - 1))
        exit(1)

    return tuple(argv[1:])


# get prefix and model name from command line
def get_model_name():
    return get_from_commandline_args(1, "'model_name'")


hist_suffix = "_hist_Graph_LSTM_output.pkl"
loss_suffix = "_scalar_loss.pkl"
loss_1_suffix = "_scalar_loss_1.pkl"

if __name__ == '__main__':
    model_name = get_model_name()[0]

    # path declarations
    data_dir = r"/mnt/HDD_data/data/tensorboard_data/"

    print("Loading files …")
    try:
        loss = pd.load(open(data_dir + model_name + loss_suffix, 'rb'))
    except IOError:
        loss = pd.load(open(data_dir + model_name + loss_1_suffix, 'rb'))
    print("done.")

    print("Plotting training loss …")
    plot_helper.plot_loss(loss.values, savepath=model_name + "_training_loss")
    print("done.")
    # plot_helper.plot_distribution(a, "Graph LSTM", savepath=data_dir + model_name + "glstm_output")
    # plot_helper.plot_histogram_continuous(a, "Graph LSTM", data_epochs=100)#, savepath="/home/matthias/testdist2")
