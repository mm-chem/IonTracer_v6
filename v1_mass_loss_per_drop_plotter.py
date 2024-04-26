import math
import pickle
import numpy as np
from scipy import signal as sig
from scipy import stats
import matplotlib.pyplot as plt
import os
from tkinter import filedialog as fd
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter


def generate_filelist(folder, termString):
    # NOTE: folders variable must be a list, even if it is a list of one
    filelist = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(termString):
                filelist.append(os.path.join(root, file))
                print("Loaded: " + filelist[-1])
    return filelist

def gauss(x, A, mu, sigma, offset):
    return offset + A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def plotter(folder):
    folder = folder.rsplit('.', maxsplit=1)[0] + ".pickled"
    files = generate_filelist(folder, '_mass_loss_per_drop.pickle')
    analysis_name = folder.rsplit('.', maxsplit=1)[0]
    new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]
    analysis_name = analysis_name + '.figures/'
    try:
        os.mkdir(analysis_name)
    except FileExistsError:
        print("Path exists already.")
    analysis_name = analysis_name + '/' + new_folder_name

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

    mass_change_dist = []
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for point in db:
            mass_change_dist.append(point)
        dbfile.close()

    st_dev = np.std(mass_change_dist)
    st_error = st_dev / np.sqrt(len(mass_change_dist))
    print("Std_error: " + str(st_error))

    fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
    hist_out = ax.hist(mass_change_dist, 100, range=[-10000, 30000], color='maroon')

    bins = hist_out[1][0:-1]
    counts = hist_out[0]

    try:
        A_constraints = [40, 500]  # Amplitude
        mu_constraints = [10000, 20000]  # x-shift
        sigma_constraints = [1000, 15000]  # Width
        offset_constraints = [0, 1]  # Vertical offset
        lower_bounds = [A_constraints[0], mu_constraints[0], sigma_constraints[0], offset_constraints[0]]
        upper_bounds = [A_constraints[1], mu_constraints[1], sigma_constraints[1], offset_constraints[1]]
        param, param_cov = curve_fit(gauss, bins, np.array(counts), bounds=(lower_bounds, upper_bounds))

        peak_contrib_to_slice = gauss(bins, param[0], param[1], param[2], param[3])

        ax.plot(bins, peak_contrib_to_slice, linewidth=3, linestyle="solid", color='black')
        text_string = f'{param[1]:.2f}'
        ax.text(param[1] + 15000, param[0], text_string, fontsize=16)
        print("Amplitude Computed Peak Center: ", str(param[1]))
    except:
        print("Unable to fit amp-computed charge loss to Gaussian.")

    ax.set_title("")
    ax.set_xlabel('Mass Loss (kDa)', fontsize=30, weight='bold')
    ax.set_ylabel('Counts', fontsize=30, weight='bold')
    ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=3, length=4)
    ax.tick_params(axis='y', which='minor', width=3, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.set_xticks([-10000, -5000, 0, 5000, 10000, 15000, 20000, 25000, 30000],
                  ["-10", "-5", "0", "5", "10", "15", "20", "25", "30"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.savefig(analysis_name + '_exported_mass_loss_dist.png', bbox_inches='tight', dpi=300.0,
                transparent='true')


if __name__ == "__main__":
    folder = fd.askdirectory(title="Choose top folder")
    plotter(folder)
