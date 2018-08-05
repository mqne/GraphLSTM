# plots training data: histograms, distributions, loss

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


loss_suffix = "_scalar_loss.pkl"
loss_1_suffix = "_scalar_loss_1.pkl"

glstm_hist_suffix = "_hist_Graph_LSTM_output.pkl"
dpren_hist_suffix = "_hist_Region_Ensemble_net_output.pkl"
network_hist_suffix = "_hist_Network_output.pkl"

glstm_comp_hist_suffix = "_comp_hist_Graph_LSTM_output.pkl"
dpren_comp_hist_suffix = "_comp_hist_Region_Ensemble_net_output.pkl"
network_comp_hist_suffix = "_comp_hist_Network_output.pkl"

val_glstm_hist_suffix = "_validation_hist_Graph_LSTM_output.pkl"
val_dpren_hist_suffix = "_validation_hist_Region_Ensemble_net_output.pkl"
val_network_hist_suffix = "_validation_hist_Network_output.pkl"

if __name__ == '__main__':
    model_name = get_model_name()[0]

    # path declarations
    data_dir = r"/mnt/HDD_data/data/tensorboard_data/"

    print("Loading files …")

    # loss
    try:
        loss = pd.load(open(data_dir + model_name + loss_suffix, 'rb'))
    except IOError:
        loss = pd.load(open(data_dir + model_name + loss_1_suffix, 'rb'))

    # distributions
    # glstm_dist = pd.load(open(data_dir + model_name + glstm_comp_hist_suffix, 'rb'))
    # dpren_dist = pd.load(open(data_dir + model_name + dpren_comp_hist_suffix, 'rb'))
    # network_dist = pd.load(open(data_dir + model_name + network_comp_hist_suffix, 'rb'))
#
    # # histograms
    # val_glstm_hist = pd.load(open(data_dir + model_name + val_glstm_hist_suffix, 'rb'))
    # val_network_hist = pd.load(open(data_dir + model_name + val_network_hist_suffix, 'rb'))
    # val_dpren_hist = pd.load(open(data_dir + model_name + val_dpren_hist_suffix, 'rb'))
#
    # dpren_hist = pd.load(open(data_dir + model_name + dpren_hist_suffix, 'rb'))
    glstm_hist = pd.load(open(data_dir + model_name + glstm_hist_suffix, 'rb'))
    # network_hist = pd.load(open(data_dir + model_name + network_hist_suffix, 'rb'))

    print("done.")

    # removing dots from model name for latex file compatibility
    model_name = model_name.replace('.', ',')

    # print("Plotting training loss …")
    # plot_helper.plot_loss(loss.values, savepath=model_name + "_training_loss")
    # print("done.")

    # print("Plotting distributions …")
    # plot_helper.plot_distribution(glstm_dist, "Graph LSTM", savepath=model_name + "_glstm_output", ymax=0.3)
    # plot_helper.plot_distribution(dpren_dist, "DeepPrior+REN", savepath=model_name + "_dpren_output", ymax=250, ymin=0)
    # plot_helper.plot_distribution(network_dist, "Full network", savepath=model_name + "_net_output", ymax=250, ymin=0)
    # plot_helper.plot_distribution(network_dist, "Full network", savepath=model_name + "_net_output_small", ymax=250, ymin=0, xticks=(0, 50, 100), figsize=(3,2))
    # print("done.")

    print("Plotting histograms …")
    # plot_helper.plot_histogram_continuous(val_glstm_hist, "Graph LSTM", plot_normal=True)
    # plot_helper.plot_histogram_continuous(val_network_hist, "Full network", plot_normal=True, xmax=250, xmin=-100)
    # plot_helper.plot_histogram_continuous(val_dpren_hist, "DeepPrior+REN", plot_normal=True, xmax=250, xmin=-100)

    # plot_helper.plot_histogram_continuous(network_hist, "Network", xmax=250, xmin=0,
    #                                       data_epochs=100, plot_start_epoch=40, plot_end_epoch=100,
    #                                       mm_per_unit=1)
    # plot_helper.plot_histogram_continuous(dpren_hist, "DeepPrior+REN", xmax=250, xmin=0,
    #                                       data_epochs=100, plot_start_epoch=80, plot_end_epoch=100,
    #                                       mm_per_unit=1)
    plot_helper.plot_histogram_continuous(glstm_hist, "Graph LSTM",
                                          data_epochs=100, plot_start_epoch=35, plot_end_epoch=100,
                                          xmin=-.3, xmax=.3, xticks=7,
                                          figsize=(plot_helper.PAGEWIDTH_INCHES, 3), additional_mms_to_be_evaluated=[4.13618836461],
                                          )  # savepath=model_name + "_glstm_hist")
    print("Done.")
