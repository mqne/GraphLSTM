import numpy as np
import matplotlib.pyplot as plt
from helpers import ErrorCalculator as Ec


# Define TUM corporate design colors
# Taken from http://portal.mytum.de/corporatedesign/index_print/vorlagen/index_farben
class TumColours:
    Blue = '#0065BD'
    SecondaryBlue = '#005293'
    SecondaryBlue2 = '#003359'
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


PlotColours = {
    1: TumColours.Blue,
    3: TumColours.AccentOrange,
    5: TumColours.SecondaryBlue2,
    2: TumColours.AccentBlue,
    4: TumColours.Gray,
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

    print("PyPlot font style has been set to match TUM thesis.")


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
        y = np.linspace(0, 100, len(sorted_errors) + 1)
        x = np.append(sorted_errors, sorted_errors[-1])
        plt.step(x, y, color=colour)

    plt.xlim(0, max_err)
    plt.ylim(0, 100)
    plt.yticks(np.linspace(0, 100, 3))
    plt.xlabel(xlabel)  # todo unit mm?
    plt.ylabel(ylabel)

    if legend is not None:
        plt.legend(legend, fancybox=False, edgecolor=TumColours.LightGray)

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
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
    plot_accuracy_curve([Ec.per_frame(x) for x in make_tuple(errors)],
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
    plot_accuracy_curve([Ec.per_frame_and_joint(x).flatten() for x in make_tuple(individual_errors)],
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
    plot_accuracy_curve([np.amax(Ec.per_frame_and_joint(x), axis=1) for x in make_tuple(individual_errors)],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        legend=legend,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colours=colours)


def make_tuple(o):
    return (o,) if not isinstance(o, (tuple, list)) else tuple(o)


set_thesis_style()


def main():
    import sys

    raise NotImplementedError


if __name__ == "__main__":
    main()
