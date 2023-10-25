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

    files = generate_filelist('.pickle')
    slope_distributions = []
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        slope_distributions.append(db)
        dbfile.close()

    mass_collection =
    # Plot a 2D mass spectrum
    # Rayleigh line parameters are in DIAMETER (nm)
    plot_rayleigh_line(axis_range=[0, 200])
    heatmap, xedges, yedges = np.histogram2d(mass_collection, charge_collection, bins=[160, 120],
                                             range=[[min(hist_mass_bins), max(hist_mass_bins)],
                                                    [min(hist_charge_bins), max(hist_charge_bins)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    gaussmap = gaussian_filter(heatmap, 1, mode='nearest')

    # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
    fig, ax = plt.subplots(layout='tight')
    ax.imshow(gaussmap.T, cmap='nipy_spectral_r', extent=extent, origin='lower', aspect='auto',
              interpolation='none')

    ax.set_title("")
    ax.set_xlabel('Mass (MDa)', fontsize=24, weight='bold')
    ax.set_ylabel('Charge', fontsize=24, weight='bold')
    ax.set_xticks(hist_mass_bins, hist_mass_labels)
    # ax.set_yticks(hist_charge_bins, hist_charge_labels)
    ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=3, length=4)
    ax.tick_params(axis='y', which='minor', width=3, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    # x = np.multiply(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]), 1000000)
    # labels = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "150"]

    if save_plots:
        plt.savefig(str(analysis_name) + '_mass_spectrum_2D.png', bbox_inches='tight', dpi=300.0, pad_inches=0.5,
                    transparent='true')
    if show_plots:
        plt.show()
    else:
        plt.close()