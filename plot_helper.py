import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from helpers import ErrorCalculator as Ec
from helpers import HAND_GRAPH_HANDS2017_INDEX_DICT, reverse_dict


# Define TUM corporate design colors
# Taken from http://portal.mytum.de/corporatedesign/index_print/vorlagen/index_farben
class TumColours:
    Blue = '#0065BD'
    SecondaryBlue = '#005293'
    SecondaryBlue_20 = '#ccdce9'
    SecondaryBlue_50 = '#7fa8c9'
    SecondaryBlue_80 = '#3375a9'
    SecondaryBlue2 = '#003359'
    SecondaryBlue2_20 = '#ccd6de'
    SecondaryBlue2_50 = '#7f99ac'
    SecondaryBlue2_80 = '#335c7a'
    Black = '#000000'
    White = '#FFFFFF'
    DarkGray = '#333333'
    Gray = '#808080'
    LightGray = '#CCCCC6'
    AccentGray = '#DAD7CB'
    AccentOrange = '#E37222'
    AccentGreen = '#A2AD00'
    AccentLightBlue = '#98C6EA'
    AccentBlue = '#64A0C8'
    XSecondaryBlue_10 = '#e5edf4'
    XSecondaryBlue_35 = '#a6c2d9'
    XSecondaryBlue_65 = '#598eb9'


PlotColours = {
    1: TumColours.Blue,
    3: TumColours.AccentOrange,
    5: TumColours.SecondaryBlue2,
    2: TumColours.AccentBlue,
    4: TumColours.Gray,
    6: TumColours.SecondaryBlue_20,
    7: TumColours.SecondaryBlue_80,
    8: TumColours.LightGray,
    9: TumColours.DarkGray,
    10: TumColours.SecondaryBlue2_50
}


def set_thesis_style():
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage[T1]{fontenc}'
                                  r'\usepackage[sc]{mathpazo}'
                                  r'\linespread{1.05}'
                                  r'\usepackage[utf8]{inputenc}'
                                  r'\usepackage[final]{microtype}'
           )
    plt.rc('font', family='serif')
    plt.rc('pgf', rcfonts=False)
    plt.rc('figure', autolayout=True)

    plt.rc('lines', linewidth=0.625)

    print("PyPlot font style has been set to match TUM thesis.")


def plot_loss(values, smoothing=.03, polyorder=3, name=None, epochs=100, xlabel="Epoch", ylabel="Training loss", savepath=None, figsize=(5, 3), fontsize=10,
              colour=TumColours.SecondaryBlue, bg_colour=TumColours.SecondaryBlue_20):
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)

    x = np.linspace(0, epochs, len(values))

    plt.plot(x, values, color=bg_colour)
    # scale Savitzky-Golay window length to data
    window_length = int(smoothing * len(values))
    # make window length odd number
    window_length += 1 - window_length % 2
    smoothed_values = savgol_filter(values, window_length, polyorder)
    plt.plot(x, smoothed_values, color=colour)

    plt.xlim(0, epochs)
    plt.ylim(0, 2)

    plt.xticks(np.linspace(0, epochs, 11))
    plt.yticks(np.linspace(0, 2, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if name is not None:
        legend = (name + " (actual)", name + " (smoothed)")
        plt.legend(legend, fancybox=False, edgecolor=TumColours.LightGray)

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
        plt.savefig(savepath + ".png", dpi=300)
    else:
        plt.show()
    plt.close()
    plt.rc('font', size=plt.rcParamsDefault['font.size'])


def plot_distribution(compressed_histogram, name, data_epochs=100, plot_epochs=None, xticks=None, ylim=0.5,
                      xlabel="Epoch", ylabel="output", savepath=None, figsize=(5, 3), fontsize=10):
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)

    x = np.asarray(compressed_histogram.steps) * data_epochs / compressed_histogram.steps[-1]
    ylabel = name + ' ' + ylabel
    if plot_epochs is None:
        plot_epochs = data_epochs
    if xticks is None:
        if plot_epochs % 10 != 0:
            print("WARNING: automatic x-ticks in distribution for %i epochs yields non-integer labels.\n"
                  "Consider passing manual x-ticks via parameter 'xticks'." % plot_epochs)
        xticks = np.linspace(0, plot_epochs, 11)

    plt.fill_between(x, compressed_histogram.Infm, compressed_histogram.Infp,
                     color=TumColours.SecondaryBlue_20, linewidth=0.0)
    plt.fill_between(x, compressed_histogram.stdm3, compressed_histogram.stdp3,
                     color=TumColours.XSecondaryBlue_35, linewidth=0.0)
    plt.fill_between(x, compressed_histogram.stdm2, compressed_histogram.stdp2,
                     color=TumColours.SecondaryBlue_50, linewidth=0.0)
    plt.fill_between(x, compressed_histogram.stdm1, compressed_histogram.stdp1,
                     color=TumColours.XSecondaryBlue_65, linewidth=0.0)
    plt.plot(x, compressed_histogram.median,
             color=TumColours.SecondaryBlue)

    plt.xlim(0, plot_epochs)
    plt.ylim(-ylim, ylim)

    plt.xticks(xticks)
    plt.yticks(np.linspace(-ylim, ylim, 3))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
        plt.savefig(savepath + ".png", dpi=300)
    else:
        plt.show()
    plt.close()
    plt.rc('font', size=plt.rcParamsDefault['font.size'])


# TODO: plot histograms


# plot cumulative error across validation frames
def plot_accuracy_curve(individual_errors, xlabel, ylabel, legend=None, savepath=None, figsize=(5, 3), max_err=40, fontsize=10,
                        colours=None):
    individual_errors = make_tuple(individual_errors)
    if colours is not None:
        colours = make_tuple(colours)

    # checking if all error entries have same length
    if len(individual_errors) > 1:
        # prepare colours
        if colours is None:
            if len(individual_errors) > len(PlotColours):
                raise ValueError("Must define custom colours for len(individual_errors) > len(PlotColours).")
        else:
            if len(colours) != len(individual_errors):
                raise ValueError("You need to provide len(individual_errors) = %i colours, but found %i"
                                 % (len(individual_errors), len(colours)))

        first_length = len(individual_errors[0])
        for i in range(1, len(individual_errors)):
            if len(individual_errors[i]) != first_length:
                print("WARNING: Plotting accuracy curve for arrays of different lengths")
                break

    if colours is None:
        colours = list([PlotColours[k] for k in sorted(PlotColours.keys())])[:len(individual_errors)]

    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)

    for err, colour in zip(individual_errors, colours):
        sorted_errors = np.sort(err)
        # prevent memory exhaustion
        sorted_errors = sorted_errors[::len(sorted_errors) // 10000 + 1]
        y = np.linspace(0, 100, len(sorted_errors) + 1)
        x = np.append(sorted_errors, sorted_errors[-1])
        plt.step(x, y, color=colour)

    plt.xlim(0, max_err)
    plt.ylim(0, 100)
    plt.yticks(np.linspace(0, 100, 3))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend is not None:
        plt.legend(legend, fancybox=False, edgecolor=TumColours.LightGray)

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
        plt.savefig(savepath + ".png", dpi=300)
    else:
        plt.show()
    plt.close()
    plt.rc('font', size=plt.rcParamsDefault['font.size'])
    # more potentially interesting rc values: lines.linewidth=1.5, lines.markersize=6.0
    # for more see plt.rcParams


def plot_average_frame_error(errors, xlabel=r'Average Joint Error (mm)', ylabel=r'No.\ of Frames (\%)',
                             legend=None,
                             savepath=None, figsize=(5, 3), max_err=40, fontsize=10,
                             colours=None):
    plot_accuracy_curve([Ec.per_frame(x) for x in errors],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        legend=legend,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colours=colours)


# standard error metric 2 from hands2017 paper
def plot_ratio_of_joints_within_bound(individual_errors, xlabel=r'Joint Error (mm)', ylabel=r'No.\ of Joints (\%)',
                                      legend=None,
                                      savepath=None, figsize=(5, 3), max_err=40, fontsize=10,
                                      colours=None):
    plot_accuracy_curve([Ec.per_frame_and_joint(x).flatten() for x in individual_errors],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        legend=legend,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colours=colours)


# standard error metric 3 from hands2017 paper
def plot_ratio_of_frames_with_all_joints_within_bound(individual_errors, xlabel=r'Maximum Joint Error (mm)',
                                                      legend=None,
                                                      ylabel=r'No.\ of Frames (\%)', savepath=None, figsize=(5, 3),
                                                      max_err=40,
                                                      fontsize=10,
                                                      colours=None):
    plot_accuracy_curve([np.amax(Ec.per_frame_and_joint(x), axis=1) for x in individual_errors],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        legend=legend,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colours=colours)


# positions of joints in the violin plot when grouped by joint type (wrist, MCP, PIP, DIP, TIP)
joint_pos_group_by_type = {'Wrist': 1,
                           'TMCP': 3, 'IMCP': 4, 'MMCP': 5, 'RMCP': 6, 'PMCP': 7,
                           'TPIP': 9, 'IPIP': 10, 'MPIP': 11, 'RPIP': 12, 'PPIP': 13,
                           'TDIP': 15, 'IDIP': 16, 'MDIP': 17, 'RDIP': 18, 'PDIP': 19,
                           'TTIP': 21, 'ITIP': 22, 'MTIP': 23, 'RTIP': 24, 'PTIP': 25,
                           }

# positions of joints in the violin plot when grouped by finger (wrist, thumb, index, middle, ring, pinkie)
joint_pos_group_by_finger = {"Wrist": 1,
                             "TMCP": 3, "IMCP": 8, "MMCP": 13, "RMCP": 18, "PMCP": 23,
                             "TPIP": 4, "TDIP": 5, "TTIP": 6,
                             "IPIP": 9, "IDIP": 10, "ITIP": 11,
                             "MPIP": 14, "MDIP": 15, "MTIP": 16,
                             "RPIP": 19, "RDIP": 20, "RTIP": 21,
                             "PPIP": 24, "PDIP": 25, "PTIP": 26}


def violinplot_error_per_joint(individual_errors,
                               index_dict=HAND_GRAPH_HANDS2017_INDEX_DICT,
                               group_by_joint_type=None,
                               ylabel='Error in mm',
                               max_err=20,
                               savepath=None,
                               figsize=(8, 3),
                               fontsize=10):
    errors_per_joint = Ec.reduce_xyz_norm(individual_errors)
    node_dict = reverse_dict(index_dict)
    labels = [node_dict[i] for i in range(len(index_dict))]

    # For grouping, other joint constellations than hands2017 are not supported in this implementation
    if group_by_joint_type is not None:
        if index_dict != HAND_GRAPH_HANDS2017_INDEX_DICT:
            raise NotImplementedError("Grouped violin plot is only implemented for hands2017-style indices")
        if group_by_joint_type:
            pos_dict = joint_pos_group_by_type
        else:
            pos_dict = joint_pos_group_by_finger
        pos = [pos_dict[node_dict[i]] for i in range(len(node_dict))]
    else:
        pos = np.arange(1, len(labels) + 1)

    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)
    sp = plt.subplot()

    vp = plt.violinplot(errors_per_joint, pos, showmeans=True, showextrema=False, points=1000)
    for p in vp['bodies']:
        p.set_facecolor(TumColours.SecondaryBlue2_20)
        p.set_alpha(1)
    vp['cmeans'].set_color(TumColours.SecondaryBlue)
    vp['cmeans'].set_alpha(1)

    sp.set_xticks(pos)
    sp.set_xticklabels(labels, rotation=45)

    plt.ylim(0, max_err)
    plt.ylabel(ylabel)

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
        plt.savefig(savepath + ".png", dpi=300)
    else:
        plt.show()
    plt.close()
    plt.rc('font', size=plt.rcParamsDefault['font.size'])


def make_tuple(o):
    return (o,) if not isinstance(o, (tuple, list)) else tuple(o)


set_thesis_style()


def main():
    import sys

    raise NotImplementedError


if __name__ == "__main__":
    main()
