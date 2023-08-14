import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd
import v6_STFT_analysis as STFT
import v1_DropShot as TraceHandler


class ZxxBackdrop:
    def __init__(self, trace_folder):
        self.Zxx = None
        self.f_range_offset = None
        self.t_range_offset = None
        self.resolution = None
        self.Zxx_filename = None
        self.daughter_traces = []
        self.trace_folder = trace_folder
        self.load_Zxx()

    def find_Zxx_file(self):
        for root, dirs, files in os.walk(self.trace_folder):
            for file in files:
                if file.endswith(".Zxx"):
                    self.Zxx_filename = os.path.join(root, file)

    def load_Zxx(self):
        self.find_Zxx_file()
        with open(self.Zxx_filename, newline='') as file:
            Zxx_string = file.read().replace('\n', '')
        Zxx_string = Zxx_string.split("//, ")
        header_data = Zxx_string[0]
        header_data = header_data.split(', ')[0]
        header_data = header_data.split('|')
        self.f_range_offset = float(header_data[0])
        self.resolution = float(header_data[1])
        self.t_range_offset = float(header_data[2])
        Zxx_string = Zxx_string[1]
        Zxx_string = Zxx_string.replace('[', ',')
        Zxx_string = Zxx_string.replace(']', ',')
        Zxx_string = Zxx_string.split(',')
        Zxx_array = []
        for element in Zxx_string:
            if len(element) > 2:
                element = element.replace(' ', ', ')
                element = element.replace(', , ', ', ')
                element = element.split(',')
                Zxx_array.append([float(i) for i in element])
        self.Zxx = np.flipud(np.rot90(np.array(Zxx_array)))

    def plot_ion_trace_on_Zxx(self, drop, trace_ID_str, save_plots=False):
        try:
            plt.plot(np.array(drop.trace_indices), np.array(drop.fundamental_trace), color='red')
            plt.plot(np.array(drop.trace_indices), np.array(drop.trace_harm),
                     linestyle='dashdot', color='red')
        except Exception as e:
            print('Trace plotting error:', e)

        plot_steps = len(self.Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
        plot_height = len(self.Zxx)

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[
            slice(self.f_range_offset, plot_height * self.resolution + self.f_range_offset, self.resolution),
            slice(self.t_range_offset, plot_steps + self.t_range_offset, 1)]
        plt.pcolormesh(x, y, self.Zxx)
        plt.colorbar()
        if save_plots:
            plt.savefig(str(self.Zxx_filename) + '_' + str(trace_ID_str) + '_zxx.png')
            plt.close()
        else:
            plt.show()

    def plot_all_traces_on_Zxx(self, min_freq, max_freq, include_harmonics=False, plot_trace_overlay=True, save_file_ID='test_trace_file', save_plots=False):
        try:
            if plot_trace_overlay:
                if include_harmonics:
                    for trace in self.daughter_traces:
                        if np.max(trace.trace_harm) < max_freq and np.min(trace.trace) > min_freq:
                            plt.plot(np.array(trace.trace_indices), np.array(trace.trace), color='magenta')
                            plt.plot(np.array(trace.trace_indices), np.array(trace.trace_harm), linestyle='dashdot', color='magenta')
                else:
                    for trace in self.daughter_traces:
                        if max_freq > np.max(trace.trace) and np.min(trace.trace) > min_freq:
                            if len(trace.fragments) == 1:
                                plt.plot(np.array(trace.trace_indices), np.array(trace.trace), color='magenta')
                            else:
                                for frag in trace.fragments:
                                    plt.plot(np.array(frag.trace_indices), np.array(frag.trace))


        except Exception as e:
            print('Trace plotting error:', e)

        plot_steps = len(self.Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
        plot_height = len(self.Zxx)
        y_vals = range(int(self.f_range_offset), int(plot_height * self.resolution + self.f_range_offset),
                       int(self.resolution))

        min_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - min_freq))
        max_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - max_freq))


        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[
            slice(y_vals[min_freq_index], y_vals[max_freq_index], self.resolution),
            slice(self.t_range_offset, plot_steps + self.t_range_offset, 1)]
        plt.pcolormesh(x, y, self.Zxx[0:-1][min_freq_index:max_freq_index], cmap='hot')
        plt.colorbar()
        if save_plots:
            plt.savefig(str(self.Zxx_filename) + '_' + str(save_file_ID) + '_zxx.png')
            plt.close()
        else:
            plt.show()

    def plot_zoomed_drop_on_Zxx(self, drop, drop_index, trace_ID_str, save_folder, save_plots=False):
        # Determine max and min traces for the fundamental traces
        min_fundamental_trace = 10000000
        max_fundamental_trace = 0
        min_harmonic_trace = 0
        max_harmonic_trace = 0
        if drop.fundamental_trace[-1] < min_fundamental_trace:
            min_fundamental_trace = drop.fundamental_trace[-1]
            min_harmonic_trace = 2 * min_fundamental_trace
        if drop.fundamental_trace[-1] > max_fundamental_trace:
            max_fundamental_trace = drop.fundamental_trace[-1]
            max_harmonic_trace = 2 * max_fundamental_trace

        # Generate zoomed figure for fundamentals
        min_freq = min_fundamental_trace - 2000
        max_freq = max_fundamental_trace + 500
        try:
            plt.plot(np.array(drop.trace_indices), np.array(drop.fundamental_trace), color='magenta')
            plt.axvline(x=drop_index, color="green")
            text_string = f'{drop.freq_change_magnitude:.2f} Hz'
            plt.text(drop_index - 30, max_fundamental_trace - 200, text_string, color="magenta")
            text_string = f'{drop.freq_computed_charge_loss:.2f} Charge'
            plt.text(drop_index - 30, max_fundamental_trace - 300, text_string, color="magenta")
        except Exception as e:
            print('Trace plotting error:', e)

        try:
            plot_steps = len(self.Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
            plot_height = len(self.Zxx)
            y_vals = range(int(self.f_range_offset), int(plot_height * self.resolution + self.f_range_offset),
                        int(self.resolution))
            min_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - min_freq))
            max_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - max_freq))

            # generate 2 2d grids for the x & y bounds
            y, x = np.mgrid[
                slice(y_vals[min_freq_index], y_vals[max_freq_index], self.resolution),
                slice(self.t_range_offset, plot_steps + self.t_range_offset, 1)]
            plt.pcolormesh(x, y, self.Zxx[0:-1][min_freq_index:max_freq_index], cmap='hot')
            plt.colorbar()
            if save_plots:
                plt.savefig(save_folder + '/drx_' + trace_ID_str + '.png')
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print("Background plotting error", e)

    def plot_zoomed_trace_on_Zxx(self, trace, trace_ID_str, save_folder, save_plots=False):
        # Determine max and min traces for the fundamental traces
        plot_trace_on_Zxx = True
        min_fundamental_trace = 10000000
        max_fundamental_trace = 0
        min_harmonic_trace = 0
        max_harmonic_trace = 0
        if trace.trace[-1] < min_fundamental_trace:
            min_fundamental_trace = trace.trace[-1]
            min_harmonic_trace = 2 * min_fundamental_trace
        if trace.trace[-1] > max_fundamental_trace:
            max_fundamental_trace = trace.trace[-1]
            max_harmonic_trace = 2 * max_fundamental_trace

        # Generate zoomed figure for fundamentals
        min_freq = min_fundamental_trace - 2000
        max_freq = max_fundamental_trace + 500
        try:
            if plot_trace_on_Zxx:
                plt.plot(np.array(trace.trace_indices), np.array(trace.trace), color='magenta')
        except Exception as e:
            print('Trace plotting error:', e)

        try:
            plot_steps = len(self.Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
            plot_height = len(self.Zxx)
            y_vals = range(int(self.f_range_offset), int(plot_height * self.resolution + self.f_range_offset),
                        int(self.resolution))
            min_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - min_freq))
            max_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - max_freq))

            # generate 2 2d grids for the x & y bounds
            y, x = np.mgrid[
                slice(y_vals[min_freq_index], y_vals[max_freq_index], self.resolution),
                slice(self.t_range_offset, plot_steps + self.t_range_offset, 1)]
            plt.pcolormesh(x, y, self.Zxx[0:-1][min_freq_index:max_freq_index], cmap='hot')
            plt.colorbar()

            for drop in trace.drops:
                text_string = f'{drop.freq_computed_charge_loss:.2f}'
                plt.text(drop.drop_index - 10, drop.finalFreq - 200, text_string, color="white")

            if save_plots:
                plt.savefig(save_folder + '/trx_' + trace_ID_str + '_' + trace.file + '.png')
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print("Background plotting error", e)

if __name__ == "__main__":
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    SPAMM = 2
    drop_threshold = -10
    before_existence_threshold = 15
    after_existence_threshold = 15

    win = tk.Tk()
    win.focus_force()
    win.withdraw()
    folder = fd.askdirectory(title="Choose traces folder")
    print(folder)
    filelists = STFT.generate_filelist([folder], ".trace")
    file_count = len(filelists[0])
    analysis_name = folder.rsplit('.', maxsplit=1)[0]
    new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]

    file_counter = 0
    ZxxFoundation = ZxxBackdrop(filelists[0][0].rsplit('\\', 1)[0])
    for file in filelists[0]:
        file_counter = file_counter + 1
        print("Processing file " + str(file_counter) + " of " + str(file_count))
        newTrace = TraceHandler.Trace(file, SPAMM, drop_threshold=drop_threshold)
        ZxxFoundation.daughter_traces.append(newTrace)

    trace_counter = 0
    for trace in ZxxFoundation.daughter_traces:
        trace_counter += 1
        print('Trace ' + str(trace_counter) + ': '
              + str(trace.trace[0]) + ' Hz Start --- '
              + str(trace.avg_slope)
              + ' Hz/s Drift' + ' --- Avg Mass: ' + str(trace.avg_mass) + ' Da'
              + ' --- Avg Charge: ' + str(trace.avg_charge))

    ZxxFoundation.plot_all_traces_on_Zxx(5000, 20000, plot_trace_overlay=False, include_harmonics=False)