import pickle
import numpy as np
from scipy import signal as sig
from scipy import stats
import matplotlib.pyplot as plt
import os
from tkinter import filedialog as fd


def generate_filelist(termString):
    # NOTE: folders variable must be a list, even if it is a list of one
    filelist = []
    folder = fd.askdirectory(title="Choose top folder")
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(termString):
                filelist.append(os.path.join(root, file))
                print("Loaded: " + filelist[-1])
    return filelist


if __name__ == "__main__":
    SMALL_SIZE = 18
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    file_order = [1, 0, 2, 3]
    slope_max_small = 0.06
    slope_max_large = 15
    smooth_window = 10
    smooth_polyorder = 2

    files = generate_filelist('.pickle')
    slope_distributions = []
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        slope_distributions.append(db)
        dbfile.close()

    # DEFINE ax to be the small and ax2 to be the large numbers
    colors = ['red', 'green', 'orange', 'blue', 'purple']
    color_counter = 0
    f, (ax, ax2) = plt.subplots(1, 2, facecolor='w')
    smoothed_output = False

    small_plot_dists = []
    large_plot_dists = []

    sorted_slope_distributions = []
    for n in range(len(slope_distributions)):
        sorted_slope_distributions.append(slope_distributions[file_order[n]])

    for dist in sorted_slope_distributions:
        print(stats.mode(dist)[0])
        if stats.mode(dist)[0] < 1:
            small_plot_dists.append(dist)
        else:
            large_plot_dists.append(dist)

    ax.hist(small_plot_dists, bins=250, color=['orange', 'red', 'black'], range=[0, slope_max_small])
    labels = ["0", "0.01", "0.03", "0.05"]
    x = [0.000, 0.01, 0.03, 0.05]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax2.hist(large_plot_dists, bins=150, color='teal', range=[1, slope_max_large])
    labels = ["3", "5", "7", "9", "11", "13", "15"]
    x = [3, 5, 7, 9, 11, 13, 15]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    ax.set_xlim(0, slope_max_small)
    ax2.set_xlim(1, slope_max_large)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    # plt.suptitle("Cluster Hydration Distribution")
    # plt.xlabel('Slope: Drift (Hz) per STFT Step (5ms)')
    # plt.ylabel('Counts')
    plt.tight_layout(pad=0)
    plt.show()
