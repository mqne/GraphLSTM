import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import trapz, cumtrapz
from scipy.stats import norm
from tqdm import tqdm
from collections import namedtuple

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

# pagewidth of text in thesis, obtained by LaTeX: \printinunitsof{in}\prntlen{\textwidth}
PAGEWIDTH_INCHES = 5.78853


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


def plot_loss(values, ymax=2, smoothing=.03, polyorder=3, name=None, epochs=100, xlabel="Epoch", ylabel="Training loss", savepath=None, figsize=(5, 3), fontsize=10,
              colour=TumColours.SecondaryBlue, bg_colour=TumColours.SecondaryBlue_20):
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)

    x = np.linspace(0, epochs, len(values))

    plt.plot(x, values, color=bg_colour, linewidth=plt.rcParams['lines.linewidth']/2)
    # scale Savitzky-Golay window length to data
    window_length = int(smoothing * len(values))
    # make window length odd number
    window_length += 1 - window_length % 2
    smoothed_values = savgol_filter(np.minimum(values, ymax+2), window_length, polyorder)
    plt.plot(x, smoothed_values, color=colour)

    plt.xlim(0, epochs)
    plt.ylim(0, ymax)

    plt.xticks(np.linspace(0, epochs, 11))
    plt.yticks(np.linspace(0, ymax, 5))
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


def plot_distribution(compressed_histogram, name, data_epochs=100, plot_epochs=None, xticks=None, ymax=0.5, ymin=None,
                      xlabel="Epoch", ylabel="output", savepath=None, figsize=(5, 3), fontsize=10):
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)

    ylabel = name + ' ' + ylabel
    if plot_epochs is None:
        plot_epochs = data_epochs
    if xticks is None:
        if plot_epochs % 10 != 0:
            print("WARNING: automatic x-ticks in distribution for %i epochs yields non-integer labels.\n"
                  "Consider passing manual x-ticks via parameter 'xticks'." % plot_epochs)
        xticks = np.linspace(0, plot_epochs, 11)
    if ymin is None:
        ymin = -ymax

    x = np.asarray(compressed_histogram.steps) * data_epochs / compressed_histogram.steps[-1]

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
    plt.ylim(ymin, ymax)

    plt.xticks(xticks)
    plt.yticks(np.unique((ymin, 0, ymax)))
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


# shows tensorflow bin discretization
def plot_histogram_discrete_sampled(histogram, name, data_epochs=1, plot_start_epoch=None, plot_end_epoch=None):
    print("WARNING: plot_histogram_discrete_sampled displays step artifacts from TensorFlow storage method.\n"
          "plot_histogram_continuous_sampled is more likely what you are looking for.")
    if plot_start_epoch is None:
        plot_start_epoch = 1
    if plot_end_epoch is None:
        plot_end_epoch = data_epochs
    start_index = (-1 + plot_start_epoch) * len(histogram.steps) // data_epochs
    end_index = (-1 + plot_end_epoch + 1) * len(histogram.steps) // data_epochs
    # sample histogram
    values = []
    for hs in tqdm(range(start_index, end_index), desc="Sampling uniform distributions for histogram", leave=False, unit=' histslices'):
        for i in range(len(histogram.steps[hs].bucket_low_limit)-2):
            values.extend(np.random.uniform(histogram.steps[hs].bucket_low_limit[i],
                                            histogram.steps[hs].bucket_low_limit[i+1],
                                            int(histogram.steps[hs].bucket_filling[i+1])))

    h = np.histogram(values, 'auto', density=True)
    plt.fill_between(h[1], np.insert(h[0], 0, 0.), step='pre')
    plt.show()
    plt.close()


# highest quality, but slowest
def plot_histogram_continuous(histogram, name, xmin=-0.5, xmax=0.5, xticks=11, ymax=None, mm_per_unit=137.7863735578566,
                              data_epochs=1, plot_start_epoch=None, plot_end_epoch=None, plot_normal=False,
                              additional_mms_to_be_evaluated=tuple(),
                              xlabel="output", ylabel="Prediction density", savepath=None, figsize=(5, 2),
                              fontsize=10):
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)

    if plot_start_epoch is None:
        plot_start_epoch = 1
    if plot_end_epoch is None:
        plot_end_epoch = data_epochs
    start_index = (-1 + plot_start_epoch) * len(histogram.steps) // data_epochs
    end_index = (-1 + plot_end_epoch + 1) * len(histogram.steps) // data_epochs

    # estimate histogram by summing gaussians
    num_samples = 1000
    x = np.linspace(xmin, xmax, num_samples)
    y = np.zeros(num_samples)
    for hs in tqdm(range(start_index, end_index), desc="Calculating Gaussians for histogram", leave=False, unit=' histslices'):
        for i in range(len(histogram.steps[hs].bucket_low_limit) - 2):
            y_norm = norm.pdf(x,
                              np.mean((histogram.steps[hs].bucket_low_limit[i],
                                       histogram.steps[hs].bucket_low_limit[i + 1])),
                              (histogram.steps[hs].bucket_low_limit[i + 1] - histogram.steps[hs].bucket_low_limit[i]) / 1.5,)
            y += y_norm * histogram.steps[hs].bucket_filling[i + 1]
    y /= trapz(y, x)

    # divide into separate regions by standard deviation
    mean = np.average(x, weights=y)
    var = np.average((np.array(x) - mean) ** 2, weights=y)
    std = np.sqrt(var)

    # put x,y pairs in lists corresponding to their distance in standard deviations from the mean for plotting
    Xy = namedtuple('Xy', 'x,y')
    meanxy = Xy(mean, np.interp(mean, x, y))
    std05 = Xy([], [])
    std10 = Xy([], [])
    std15 = Xy([], [])
    inf = Xy([], [])

    for xi, yi in zip(x, y):
        d = np.abs(mean - xi)
        if d <= 0.5*std:
            std05.x.append(xi)
            std05.y.append(yi)
        if d <= std:
            std10.x.append(xi)
            std10.y.append(yi)
        if d <= 1.5*std:
            std15.x.append(xi)
            std15.y.append(yi)
        inf.x.append(xi)
        inf.y.append(yi)

    y_sum = cumtrapz(y, x)
    i_mean = None
    for i in range(len(x)):
        if x[i] >= mean:
            i_mean = i
            break
    y_sum_pos_part = y_sum[i_mean:]
    y_sum_pos_part -= y_sum_pos_part[0]
    y_sum_neg_part = y_sum[:i_mean][::-1]
    y_sum_neg_part = list(- y_sum_neg_part + y_sum_neg_part[0])
    y_sum_neg_part.insert(0, 0)
    if len(y_sum_neg_part) < len(y_sum_pos_part):
        y_sum_dist = y_sum_pos_part.copy()
        y_sum_dist[:len(y_sum_neg_part)] += y_sum_neg_part
        y_sum_dist[len(y_sum_neg_part):] = [1] * len(y_sum_dist[len(y_sum_neg_part):])
    else:
        y_sum_dist = y_sum_neg_part.copy()
        y_sum_dist[:len(y_sum_pos_part)] += y_sum_pos_part
        y_sum_dist[len(y_sum_pos_part):] = [1] * len(y_sum_dist[len(y_sum_pos_part):])

    # now y_sum_dist contains the fraction of samples between mean and index i at each index

    # gather information about the distribution
    result_text = list()

    result_text.append(name + " output, epochs " + str(plot_start_epoch) + " to " + str(plot_end_epoch))
    result_text.append('')
    result_text.append("1 raw unit ~ " + str(mm_per_unit) + " mm")
    result_text.append("Mean:   " + str(mean * mm_per_unit) + "\tmm, in raw units: " + str(mean))
    result_text.append("1 std ~ " + str(std * mm_per_unit) + "\tmm, in raw units: " + str(std))
    result_text.append('')

    index_step_distance = (xmax - xmin) / num_samples
    # samples within x standard deviations
    for stdfrac in (0.5, 1., 1.5, 2., 2.5, 3., 4., 5.):
        perc = np.interp(stdfrac * std * mm_per_unit,
                             np.array(list(range(len(y_sum_dist)))) * index_step_distance * mm_per_unit, np.array(y_sum_dist) * 100)
        result_text.append("Samples within\t" + str(stdfrac) + " std:\t" + str(perc) + " %")
    result_text.append('')
    # samples within x millimetres
    for mms in (1, 2, 5, 10, 15, 20, 30, 50, 100):
        perc = np.interp(mms,
                             np.array(list(range(len(y_sum_dist)))) * index_step_distance * mm_per_unit, np.array(y_sum_dist) * 100)
        result_text.append("Samples within\t" + str(mms) + " mm:\t\t" + str(perc) + " %")
    result_text.append('')
    # distance within which x percent of samples lie
    for perc in (1, 2, 3, 5, 10, 20, 25, 50, 80, 90):
        distance = np.interp(perc/100, y_sum_dist, np.array(list(range(len(y_sum_dist))))) * index_step_distance * mm_per_unit
        result_text.append(str(perc) + " %\tof values lie within\t" + str(distance) + "\tmm of the mean")
    # optional: samples within mms in additional_mms_to_be_evaluated
    for mms in additional_mms_to_be_evaluated:
        result_text.append('')
        perc = np.interp(mms,
                             np.array(list(range(len(y_sum_dist)))) * index_step_distance * mm_per_unit, np.array(y_sum_dist) * 100)
        result_text.append("Samples within\t" + str(mms) + " mm:\t\t" + str(perc) + " %")
    # plot regions coloured by standard deviations from mean (1, 2, 3, rest)
    plt.fill_between(inf.x, inf.y, color=TumColours.SecondaryBlue_20, linewidth=0.0, zorder=-50)
    plt.fill_between(std15.x, std15.y, color=TumColours.XSecondaryBlue_35, linewidth=0.0, zorder=-40)
    plt.fill_between(std10.x, std10.y, color=TumColours.SecondaryBlue_50, linewidth=0.0, zorder=-20)
    plt.fill_between(std05.x, std05.y, color=TumColours.XSecondaryBlue_65, linewidth=0.0, zorder=-10)

    # plot estimated corresponding normal distribution
    if plot_normal:
        y_norm = norm.pdf(x, mean, std)
        plt.plot(x, y_norm, color=TumColours.AccentGray)

    plt.xlim(xmin, xmax)
    ymin, ymax = plt.ylim(0, ymax)

    # plot line indicating the median
    plt.axvline(meanxy.x, color=TumColours.SecondaryBlue, zorder=-5)
    plt.fill_between(x, y, ymax, color='white', linewidth=0.0, zorder=0)

    plt.xticks(np.linspace(xmin, xmax, xticks))

    plt.xlabel(name + ' ' + xlabel)
    plt.ylabel(ylabel)

    if np.maximum(y[0], y[-1]) > 0.001:
        print("\nWARNING: significant Y values at plot edge detected. Mean and std evaluation and plotting might be inaccurate.\n"
              "Try plotting again with more permissive 'xmin' and/or 'xmax' limits.\n")
        result_warning = "THESE VALUES ARE INACCURATE. PLOT AGAIN WITH MORE PERMISSIVE xmin/xmax LIMITS."
        result_text.append(result_warning)
        result_text.insert(0, result_warning)

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
        plt.savefig(savepath + ".png", dpi=300)
        with open(savepath + '.txt', 'a') as f:
            for line in result_text:
                f.write(line + '\n')
    else:
        for line in result_text:
            print(line)
        plt.show()
    plt.close()
    plt.rc('font', size=plt.rcParamsDefault['font.size'])


# high quality, discretization of histogram visible
def plot_histogram_continuous_sampled(histogram, name, xmin=-0.5, xmax=0.5, xticks=11, ymax=None,
                                      data_epochs=1, plot_start_epoch=None, plot_end_epoch=None, plot_normal=False,
                                      xlabel="output", ylabel="Prediction density", savepath=None, figsize=(5, 2),
                                      fontsize=10):
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)

    if plot_start_epoch is None:
        plot_start_epoch = 1
    if plot_end_epoch is None:
        plot_end_epoch = data_epochs
    start_index = (-1 + plot_start_epoch) * len(histogram.steps) // data_epochs
    end_index = (-1 + plot_end_epoch + 1) * len(histogram.steps) // data_epochs

    # sample histogram
    values = []
    for hs in tqdm(range(start_index, end_index), desc="Sampling Gaussians for histogram", leave=False, unit=' histslices'):
        for i in range(len(histogram.steps[hs].bucket_low_limit)-2):
            values.extend(np.random.normal(np.mean((histogram.steps[hs].bucket_low_limit[i], histogram.steps[hs].bucket_low_limit[i+1])),
                                           (histogram.steps[hs].bucket_low_limit[i+1] - histogram.steps[hs].bucket_low_limit[i])/1.5,
                                           int(histogram.steps[hs].bucket_filling[i+1])))

    h = np.histogram(values, 'auto', density=True)
    plt.fill_between(h[1], np.insert(h[0], 0, 0.), step='pre', color=TumColours.SecondaryBlue_80, linewidth=0.0)

    # plot estimated corresponding normal distribution
    if plot_normal:
        mean, std = norm.fit(values)
        # mean = np.average(x_s, weights=y_s)
        # var = np.average((np.array(x_s) - mean) ** 2, weights=y_s)
        # std = np.sqrt(var)
        x_norm = np.linspace(xmin, xmax, 1000)
        y_norm = norm.pdf(x_norm, mean, std)
        plt.plot(x_norm, y_norm, color=TumColours.AccentGray)

    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)

    plt.xticks(np.linspace(xmin, xmax, xticks))

    plt.xlabel(name + ' ' + xlabel)
    plt.ylabel(ylabel)

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
        plt.savefig(savepath + ".png", dpi=300)
    else:
        plt.show()
    plt.close()
    plt.rc('font', size=plt.rcParamsDefault['font.size'])


# fast and step-less, but prone to artifacts
def plot_histogram_analytical(histogram, name, xmin=-0.5, xmax=0.5, xticks=11, ymax=None,
                              data_epochs=1, plot_start_epoch=None, plot_end_epoch=None, plot_normal=False,
                              epsilon=1e-4,
                              xlabel="output", ylabel="Prediction density", savepath=None, figsize=(5, 2), fontsize=10):
    print("WARNING: plot_histogram_analytical is fast, but prone to artifacts.")
    if plot_normal:
        print("WARNING: plot_normal of plot_histogram_analytical is slightly skewed.")
    plt.rc('font', size=fontsize)
    plt.figure(num=None, figsize=figsize)

    epsilon = (xmax-xmin) * epsilon
    if plot_start_epoch is None:
        plot_start_epoch = 1
    if plot_end_epoch is None:
        plot_end_epoch = data_epochs
    start_index = (-1 + plot_start_epoch) * len(histogram.steps) // data_epochs
    end_index = (-1 + plot_end_epoch + 1) * len(histogram.steps) // data_epochs

    x = []
    y = []
    weights = []
    for hs in tqdm(range(start_index, end_index), desc="Analysing input data for histogram", leave=False, unit=' histslices'):
        for i in range(len(histogram.steps[hs].bucket_low_limit)-2):
            x.append(np.mean((histogram.steps[hs].bucket_low_limit[i], histogram.steps[hs].bucket_low_limit[i + 1])))
            y.append(histogram.steps[hs].bucket_filling[i + 1])
            weights.append(histogram.steps[hs].bucket_low_limit[i + 1] - histogram.steps[hs].bucket_low_limit[i])

    sort_indices = np.argsort(x)
    x = np.array(x)[sort_indices]
    y = np.array(y)[sort_indices]
    weights = np.array(weights)[sort_indices]

    # merge x distances < epsilon
    i = 0
    x_s = []
    y_s = []
    w_s = []
    while i < len(x) - 1:
        if x[i+1] - x[i] > epsilon:
            x_s.append(x[i])
            y_s.append(y[i] / weights[i])
            # y_s.append(y[i] / np.maximum(weights[i], 1))
            w_s.append(weights[i])
        else:
            shift = 1
            while i+1+shift < len(x) and x[i+1+shift] - x[i] <= epsilon:
                shift += 1
            y_new = np.sum([y[j] for j in range(i, i+shift)])
            weight_new = np.sum([weights[j] for j in range(i, i+shift)])
            y_s.append(y_new / weight_new)
            # y_s.append(y_new / np.maximum(weight_new, 1))
            x_s.append(np.mean([x[j] for j in range(i, i+shift)]))
            w_s.append(weight_new)
            i += shift
        i += 1

    # fix faulty log data
    outliers = 0
    for i in range(1, len(y_s)-1):
        if y_s[i] == 0 and y_s[i-1] > 0 and y_s[i+1] > 0:
            y_s[i] = np.average([y_s[i-1], y_s[i+1]], weights=[x_s[i+1]-x_s[i], x_s[i]-x_s[i-1]])
            outliers += 1
    if outliers != 0:
        print("Removed %i (probable) zero-outliers from histogram data." % outliers)

    # normalize y axis
    y_s = np.array(y_s)
    y_s /= trapz(y_s, x_s)

    plt.fill_between(x_s, y_s, color=TumColours.SecondaryBlue_80, linewidth=0.0)
    # plt.plot(x_s, y_s, color=TumColours.SecondaryBlue_80)

    # plot estimated corresponding normal distribution
    if plot_normal:
        x_s = np.array(x_s)
        mean = np.average(x_s, weights=y_s)
        var = np.average((np.array(x_s[1:-1]) - mean) ** 2, weights=y_s[1:-1] * (x_s[2:] - x_s[:-2]))
        std = np.sqrt(var)
        x_norm = np.linspace(xmin, xmax, 1000)
        y_norm = norm.pdf(x_norm, mean, std)
        plt.plot(x_norm, y_norm, color=TumColours.AccentGray)

    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)

    plt.xticks(np.linspace(xmin, xmax, xticks))

    plt.xlabel(name + ' ' + xlabel)
    plt.ylabel(ylabel)

    if savepath is not None:
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
        plt.savefig(savepath + ".png", dpi=300)
    else:
        plt.show()
    plt.close()
    plt.rc('font', size=plt.rcParamsDefault['font.size'])


# plot cumulative error across validation frames
def plot_accuracy_curve(individual_errors, xlabel, ylabel, legend=None, savepath=None, figsize=(5, 3), max_err=40, fontsize=10,
                        colours=None, percentages_below=(1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99), mm_below=()):
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

    perc_text = []

    for err, colour in zip(individual_errors, colours):
        sorted_errors_full = np.sort(err)
        # prevent memory exhaustion
        sorted_errors = sorted_errors_full[::len(sorted_errors_full) // 10000 + 1]
        y = np.linspace(0, 100, len(sorted_errors) + 1)
        x = np.append(sorted_errors, sorted_errors[-1])
        plt.step(x, y, color=colour)
        # calculate some percentages below x
        model_text = []
        y_span = np.linspace(0, 100, len(sorted_errors_full))
        for perc in percentages_below:
            mm = np.interp(perc, y_span, sorted_errors_full)
            model_text.append('%2i' % perc + '% of prediction errors are below' + ' %f mm' % mm)
        perc_text.append(model_text)
        for mm in mm_below:
            perc = np.interp(mm, sorted_errors_full, y_span)
            model_text.append('%f' % perc + ' %\tof prediction errors are below' + ' %f mm' % mm)
        model_text.append('')

    plt.xlim(0, max_err)
    plt.ylim(0, 100)
    plt.yticks(np.linspace(0, 100, 3))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend is not None:
        plt.legend(legend, fancybox=False, edgecolor=TumColours.LightGray)
        for name, textlist in zip(legend, perc_text):
            textlist.insert(0, str(name) + ':\n')

    result_text = ''
    for model_text in perc_text:
        for line in model_text:
            result_text += line + '\n'
        result_text += '\n'

    if savepath is not None:
        with open(savepath + '.txt', 'a') as f:
            f.write(result_text)
        plt.savefig(savepath + ".pgf")
        plt.savefig(savepath + ".pdf")
        plt.savefig(savepath + ".png", dpi=300)
    else:
        print(result_text)
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
                                      colours=None,
                                      percentages_below=(1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99),
                                      mm_below=()):
    plot_accuracy_curve([Ec.per_frame_and_joint(x).flatten() for x in individual_errors],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        legend=legend,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colours=colours,
                        percentages_below=percentages_below,
                        mm_below=mm_below)


# standard error metric 3 from hands2017 paper
def plot_ratio_of_frames_with_all_joints_within_bound(individual_errors, xlabel=r'Maximum Joint Error (mm)',
                                                      legend=None,
                                                      ylabel=r'No.\ of Frames (\%)', savepath=None, figsize=(5, 3),
                                                      max_err=40,
                                                      fontsize=10,
                                                      colours=None,
                                                      percentages_below=(1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99),
                                                      mm_below=()):
    plot_accuracy_curve([np.amax(Ec.per_frame_and_joint(x), axis=1) for x in individual_errors],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        legend=legend,
                        savepath=savepath,
                        figsize=figsize,
                        max_err=max_err,
                        fontsize=fontsize,
                        colours=colours,
                        percentages_below=percentages_below,
                        mm_below=mm_below)


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
                               figsize=(PAGEWIDTH_INCHES, 2.5),
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
