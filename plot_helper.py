import numpy as np
import matplotlib.pyplot as plt
from helpers import ErrorCalculator as Ec


default_rcParams = plt.rcParams


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
def average_frame_error(individual_errors, savepath=None, figsize=(5, 3), max_err=40, fontsize=10,
                        colour=TumColours.Blue):
    sorted_error_per_frame = np.sort(Ec.per_frame(individual_errors))
    y = np.linspace(0, 100, len(sorted_error_per_frame) + 1)
    x = np.append(sorted_error_per_frame, sorted_error_per_frame[-1])
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)
    plt.step(x, y, color=colour)
    plt.xlim(0, max_err)
    plt.ylim(0, 100)
    plt.yticks(np.linspace(0, 100, 3))
    plt.xlabel(r'Average Frame Error (TODO)')  # todo
    plt.ylabel(r'No.\ of Frames (\%)')

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
    else:
        plt.show()
    plt.close()
    plt.rc('font', size=default_rcParams['font.size'])
    # more potentially interesting rc values: lines.linewidth=1.5, lines.markersize=6.0
    # for more see plt.rcParams


set_thesis_style()


def main():
    import sys

    raise NotImplementedError


if __name__ == "__main__":
    main()
