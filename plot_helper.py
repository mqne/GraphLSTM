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
# todo plot several networks at once
def plot_accuracy_curve(individual_errors, xlabel, ylabel, savepath=None, figsize=(5, 3), max_err=40, fontsize=10,
                        colour=TumColours.Blue):
    sorted_errors = np.sort(individual_errors)
    y = np.linspace(0, 100, len(sorted_errors) + 1)
    x = np.append(sorted_errors, sorted_errors[-1])
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)
    plt.step(x, y, color=colour)
    plt.xlim(0, max_err)
    plt.ylim(0, 100)
    plt.yticks(np.linspace(0, 100, 3))
    plt.xlabel(xlabel)  # todo
    plt.ylabel(ylabel)

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
                             savepath=None, figsize=(5, 3), max_err=40, fontsize=10,
                             colour=TumColours.Blue):
    plot_accuracy_curve(Ec.per_frame(errors),
                        xlabel=xlabel,
                        ylabel=ylabel,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colour=colour)


# standard error metric 2 from hands2017 paper
def plot_ratio_of_joints_within_bound(individual_errors, xlabel=r'Joint Error (mm)', ylabel=r'No.\ of Joints (\%)',
                                      savepath=None, figsize=(5, 3), max_err=40, fontsize=10,
                                      colour=TumColours.Blue):
    plot_accuracy_curve(Ec.per_frame_and_joint(individual_errors).flatten(),
                        xlabel=xlabel,
                        ylabel=ylabel,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colour=colour)


# standard error metric 3 from hands2017 paper
def plot_ratio_of_frames_with_all_joints_within_bound(individual_errors, xlabel=r'Maximum Joint Error (mm)',
                                                      ylabel=r'No.\ of Frames (\%)', savepath=None, figsize=(5, 3),
                                                      max_err=40,
                                                      fontsize=10,
                                                      colour=TumColours.Blue):
    plot_accuracy_curve(np.amax(Ec.per_frame_and_joint(individual_errors), axis=1),
                        xlabel=xlabel,
                        ylabel=ylabel,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colour=colour)


set_thesis_style()


def main():
    import sys

    raise NotImplementedError


if __name__ == "__main__":
    main()
