import numpy as np
import traceback
import matplotlib.pyplot as plt
import csv
import math
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
import scipy.signal as sig
import os
import itertools
import tkinter as tk
from tkinter import filedialog as fd
import v6_STFT_analysis as STFT
from scipy import signal as sig
from scipy import optimize as opt
import v1_TraceVisualizer as ZxxToolkit


def plot_rayleigh_line():
    coul_e = 1.6022E-19
    avo = 6.022E23
    surfacet = 0.07286
    # 0.07286 H20 at 20 C #0.05286 for NaCl split, 0.07286 for LiCl split 0.0179 for hexanes 0.0285 TOLUENE 0.050 for
    # mNBA
    surfacetalt = 0.0179
    perm = 8.8542E-12  # vacuum permittivity
    density = 999800  # g/m^3 999800 for water
    low_d, high_d, step_d = 0, 200, 0.5  # diameter range/step size in nm
    low = low_d * 1.0E-9
    high = high_d * 1.0E-9
    step = step_d * 1.0E-9
    qlist = []
    q2list = []
    mlist = []
    m_list = []
    for d in np.arange(low, high, step):
        q = (8 * math.pi * perm ** 0.5 * surfacet ** 0.5 * (d / 2) ** 1.5) / coul_e
        q2 = (8 * math.pi * perm ** 0.5 * surfacetalt ** 0.5 * (d / 2) ** 1.5) / coul_e
        qlist.append(q)
        q2list.append(q2)
        m = ((4 / 3) * math.pi * (d / 2) ** 3) * density * avo
        mlist.append(m)
        m_list.append(m)

    plt.plot(m_list, qlist, color='red', linewidth=3)


def d_to_mass(diameter, density=1):
    mass = ((4 / 3) * np.pi * (0.5 * diameter) ** 3) * density
    return mass


def mass_to_d(mass, density=0.9988):
    # Assumes density is given in g/ml
    mass_g = mass / 6.022E23
    diameter_cm = (np.cbrt((mass_g / density) * (3 / 4) * (1 / np.pi))) * 2
    diameter_nm = diameter_cm * 10.0E6
    return diameter_nm


def returnRayleighLine(min_mass, max_mass, number_of_points):
    coul_e = 1.6022E-19
    avo = 6.022E23
    surfacet = 0.07286
    # 0.07286 H20 at 20 C #0.05286 for NaCl split, 0.07286 for LiCl split 0.0179 for hexanes 0.0285 TOLUENE 0.050 for
    # mNBA
    surfacetalt = 0.0179
    perm = 8.8542E-12  # vacuum permittivity
    density = 999800  # g/m^3 999800 for water
    low_d, high_d = mass_to_d(min_mass), mass_to_d(max_mass)  # diameter range/step size in nm
    step_d = (high_d - low_d) / number_of_points
    low = low_d * 1.0E-9
    high = high_d * 1.0E-9
    step = step_d * 1.0E-9
    qlist = []
    q2list = []
    mlist = []
    m_list = []
    for d in np.arange(low, high, step):
        q = (8 * math.pi * perm ** 0.5 * surfacet ** 0.5 * (d / 2) ** 1.5) / coul_e
        q2 = (8 * math.pi * perm ** 0.5 * surfacetalt ** 0.5 * (d / 2) ** 1.5) / coul_e
        qlist.append(q)
        q2list.append(q2)
        m = ((4 / 3) * math.pi * (d / 2) ** 3) * density * avo
        mlist.append(m)
        m_list.append(m)
    return m_list, qlist


def sliding_window_diff(trace, width, step=1):
    winStartIndex = 0
    winEndIndex = 0 + width
    slidingDiff = np.zeros([int(np.floor(len(trace) / step)) - 1])
    for i in range(len(slidingDiff) - 1):
        avgBefore = np.average(trace[winStartIndex:winEndIndex])
        winStartIndex = winStartIndex + step
        winEndIndex = winEndIndex + step
        avgAfter = np.average(trace[winStartIndex:winEndIndex])
        slidingDiff[i] = avgAfter - avgBefore
    return slidingDiff


class Trace:
    def __init__(self, filePath, spamm, drop_threshold):
        self.SPAMM = spamm
        self.avg_slope = 0
        self.Zxx_background = []
        self.folder = filePath.rsplit('\\', 1)[0]
        self.file = filePath.rsplit('\\', 1)[-1]
        self.trace = []
        self.trace_magnitudes = []
        self.trace_phases = []
        self.trace_harm = []
        self.trace_magnitudes_harm = []
        self.trace_phases_harm = []
        self.fragmentation_indices = []
        self.trace_indices = []
        self.fragments = []
        self.drops = []
        self.drop_threshold = drop_threshold
        self.avg_mass = 0
        self.avg_charge = 0
        with open(filePath, newline='') as csvfile:
            traceReader = csv.reader(csvfile, delimiter=',')
            traceData = []
            for row in traceReader:
                traceData.append(row)

        # Assumes file with format HEADER, //, FREQ, //, AMP, //, FREQ_HARM, //, AMP_HARM
        # Produces traceData list: [0] = HEADER, [1] = FREQ[i], [2] = AMP[i], [3] = FREQ_HARM[i], [4] = AMP_HARM[i]
        traceData = [list(y) for x, y in itertools.groupby(traceData[0], lambda z: z == ' //') if not x]
        self.startFreq = float(traceData[0][0].split("|")[0])
        f_reso = float(traceData[0][0].split("|")[1])
        self.time_offset = float(traceData[0][0].split("|")[2])

        try:
            self.trace = [float(i) * f_reso + self.startFreq for i in traceData[1]]
            self.trace_magnitudes = [float(i) for i in traceData[2]]
            self.trace_phases = [float(i) for i in traceData[3]]
            self.trace_harm = [float(i) * f_reso + self.startFreq for i in traceData[4]]
            self.trace_magnitudes_harm = [float(i) for i in traceData[5]]
            self.trace_phases_harm = [float(i) for i in traceData[6]]
            self.trace_indices = [float(i) for i in traceData[7]]
        except Exception:
            print("Error reading phase-included file. Attempting old format...")
            try:
                self.trace = [float(i) * f_reso + self.startFreq for i in traceData[1]]
                self.trace_magnitudes = [float(i) for i in traceData[2]]
                self.trace_phases = []
                self.trace_harm = [float(i) * f_reso + self.startFreq for i in traceData[3]]
                self.trace_magnitudes_harm = [float(i) for i in traceData[4]]
                self.trace_phases_harm = []
                self.trace_indices = [float(i) for i in traceData[5]]
            except Exception:
                print("Could not read file at all.")

        try:
            self.fragment_trace()
            for frag in self.fragments:
                self.avg_mass = self.avg_mass + frag.mass
            self.avg_mass = self.avg_mass / len(self.fragments)
            for frag in self.fragments:
                self.avg_charge = self.avg_charge + frag.charge
            self.avg_charge = self.avg_charge / len(self.fragments)
        except Exception as e:
            print("Error dealing with fragmented traces: ", e)
            # traceback.print_exc()

        try:
            self.calculate_avg_slope()
        except Exception as e:
            print('Error encountered finding avg slope: ', e)

    def calculate_avg_slope(self):
        # In Hz/s drift
        slope_sum = 0
        counter = 0
        for frag in self.fragments:
            slope_sum += frag.linfitEquation.coefficients[0]
            counter += 1
        self.avg_slope = slope_sum / counter


    def bridge_fragments(self, frag1, frag2):
        gap_length = frag2.fragStart - frag1.fragEnd
        new_drop_point = frag1.fragEnd + np.floor(gap_length / 2)
        frag1.fragEnd = new_drop_point
        frag2.fragStart = new_drop_point
        initial_C_E = frag1.C_E
        final_C_E = frag2.C_E
        try:
            startFreq = frag1.linfitEquation(new_drop_point)
            endFreq = frag2.linfitEquation(new_drop_point)
            startMag = frag1.avg_mag
            endMag = frag2.avg_mag
            startCharge = frag1.charge
            endCharge = frag2.charge
            C_E_initial = initial_C_E
            C_E_final = final_C_E
            t_before = len(frag1.trace)
            t_after = len(frag2.trace)
            start_mass = frag1.mass
            end_mass = frag2.mass
            fundamental_trace = self.trace
            drop_index = frag1.fragEnd
            trace_indices = self.trace_indices
            folder = self.folder
            newDrop = Drop(startFreq, endFreq, startMag, endMag, startCharge, endCharge, C_E_initial, C_E_final,
                           t_before, t_after, start_mass, end_mass, fundamental_trace, drop_index,
                           trace_indices, folder)
            self.drops.append(newDrop)
        except Exception as e:
            print("Error bridging fragments...", e)
            traceback.print_exc()

    def plot_ion_traces(self, fragLines=None):
        if fragLines is None:
            fragLines = []
        plt.plot(self.trace_indices, self.trace)
        for frag in self.fragments:
            plt.plot(frag.trace_indices, frag.trace)
        # plt.plot(self.trace_harm)
        for line in fragLines:
            plt.axvline(x=line + self.time_offset, color="green")
        plt.show()

    def plot_ion_phases(self, fragLines=None):
        if fragLines is None:
            fragLines = []
        plt.plot(self.trace_phases)
        # plt.plot(self.trace_phases_harm)
        plt.plot(np.cumsum(np.abs(self.trace_phases)))
        for line in fragLines:
            plt.axvline(x=line, color="green")
        plt.show()

    def drop_threshold_scaler(self, freq):
        # This is a quadratic function to dynamically change the drop threshold depending on frequency
        f_scaled_noise_limit = -((freq - self.startFreq) / self.startFreq) ** 2 + self.drop_threshold
        return f_scaled_noise_limit

    def fragment_trace(self):
        f_scale_factor = self.trace[0]
        self.drop_threshold = self.drop_threshold_scaler(f_scale_factor)
        front_truncation = 5
        differentialTrace = sliding_window_diff(self.trace, 5)
        self.fragmentation_indices.append(0)
        plot = 0
        past_frag_index = 0
        min_drop_spacing = 2  # Don't make this determination here... set this as small as possible
        for index in range(min_drop_spacing, len(differentialTrace) - min_drop_spacing):
            comparisonSum = sum(differentialTrace[index - min_drop_spacing:index + min_drop_spacing])
            if comparisonSum < self.drop_threshold and index - past_frag_index >= 3 * min_drop_spacing:
                self.fragmentation_indices.append(index + min_drop_spacing)
                past_frag_index = index + min_drop_spacing
                # plot = 1

        self.fragmentation_indices.append(len(self.trace) - 1)
        # Fragment builder splits the trace up into fragments and creates fragment objects
        self.fragment_builder(front_truncation)
        # Fragment analyzer cuts out bad fragments and stitches the good ones together
        self.fragment_analyzer()
        if plot == 1:
            print(self.fragmentation_indices)
            self.plot_ion_traces(self.fragmentation_indices)
            # self.plot_ion_phases(self.fragmentation_indices)
            # plt.plot(differentialTrace)
            plt.show()

    def fragment_builder(self, front_truncation):
        for i in range(len(self.fragmentation_indices) - 1):
            fragStart = self.fragmentation_indices[i] + int(front_truncation)
            fragEnd = self.fragmentation_indices[i + 1]
            if fragEnd - fragStart > 2:
                fragTrace = self.trace[fragStart:fragEnd]
                fragIndices = self.trace_indices[fragStart:fragEnd]
                harmFragTrace = self.trace_harm[fragStart:fragEnd]
                fragAvgMag = np.average(self.trace_magnitudes[fragStart:fragEnd])
                harmFragAvgMag = np.average(self.trace_magnitudes_harm[fragStart:fragEnd])
                newFrag = Fragment(fragTrace, harmFragTrace, fragAvgMag, harmFragAvgMag, fragStart, fragEnd,
                                   self.SPAMM, fragIndices)
                self.fragments.append(newFrag)
            else:
                raise IndexError("Fragment start and end overlap")

    def fragment_analyzer(self):
        if len(self.fragments) > 1:
            useful_fragments = []
            for i in range(len(self.fragments) - 1):
                delta_x = self.fragments[i + 1].fragStart - self.fragments[i].fragEnd
                # # Should be in BINS (ideally)
                if delta_x <= 5:
                    self.bridge_fragments(self.fragments[i], self.fragments[i + 1])
                    useful_fragments.append(self.fragments[i])
                    if i == len(self.fragments) - 2:
                        # Add the last fragment too (if we are at the end of the line)
                        useful_fragments.append(self.fragments[i + 1])
            self.fragments = useful_fragments


class Fragment:
    def __init__(self, fragTrace, harmFragTrace, fragAvgMag, harmFragAvgMag, fragStart, fragEnd, spamm, trace_indices):
        self.SPAMM = spamm
        self.trace_indices = trace_indices  # Fragment indices
        self.fragStart = fragStart
        self.fragEnd = fragEnd
        self.trace = fragTrace
        self.energy_eV = None
        self.harm_trace = harmFragTrace
        self.avg_mag = fragAvgMag
        self.harm_avg_mag = harmFragAvgMag
        self.avg_freq = np.average(fragTrace)
        self.HAR = fragAvgMag / harmFragAvgMag
        self.C_E = None
        self.m_z = None
        self.mass = None
        self.charge = None
        self.linfitEquation = None
        self.x_axis = np.linspace(self.fragStart, self.fragEnd, len(self.trace))

        self.lin_fit_frag()
        self.magic()

    def lin_fit_frag(self):
        if len(self.trace) > 1:
            fit = np.polyfit(self.x_axis, self.trace, 1)
            self.linfitEquation = np.poly1d(fit)
        else:
            print("Unable to fit line of " + str(len(self.trace)))

    def magic(self):
        if self.SPAMM == 2:
            trap_V = 330  # trapping cones potential

            # Trap Potential/Ion eV/z ratio to energy calibration (all trap potentials)
            # Equation---Energy =Trap/(A*TTR^3 + B*TTR^2 + C*TTR + D)
            A = -0.24516
            B = 1.84976
            C = -4.92709
            D = 5.87484

            # HAR to TTR calibration (all trap settings)
            # Equation--- TTR = E*HAR^3 + F*HAR^2 + G*HAR + H
            E = -0.5305
            F = 4.0047
            G = -10.535
            H = 11.333

            # Raw Amplitude to Charge Calibration (via BSA calibration curve, may change/improve with more calibration data)
            # Equation--- Charge = (Raw Amplitude + J)/K
            J = 0.0000
            K = 0.91059  # OLD K VAL
            # K = 0.8030
            # K = 0.9999

            # (amp per charge) 1.6191E-6 for 250 ms zerofill, 1.6126E-6 for 125 ms zerofill
            # calibration value with filter added 1.3342E-6 (0.91059 with updated amplitude in analysis program)
            # calibration value with butterworth bandpass + old filter in series 1.4652E-6
            # calibration value * rough factor in analysis program (682500 currently) 0.999999
            # current calibration value ~0.87999 with 682500 factor in analysis
            # 12-20 calibration value 0.84799
            # 2-14 calibration value 0.81799

            # Charge correction factor constants from simulation of 20 kHz 400 us RC simulation across range of TTR
            # Equation--- Factors = 1/(L*TTR+M)
            L = -0.1876
            M = 1.3492

            # Energy (eV/z) to C-value conversion (m/z = C(E)/f^2)
            # C(E) function of both trap_V and energy (but not their ratio)
            # Equation--- C(E) = (A00+A10*Energy+A01*Trap_V+A20*Energy^2+A11*Energy*Trap_V+A02*Trap_V^2+A30*Energy^3+
            # A21*Energy^2*Trap_V+  A12*Energy*Trap_V^2+A03*Trap_V^3)^2
            A00 = 602227.450695831
            A10 = -11874.0933576314
            A01 = 15785.4833447021
            A20 = -110.631481581344
            A11 = 179.897639821121
            A02 = -80.623006669759
            A30 = -0.0729582264231568
            A21 = 0.3250825276845
            A12 = -0.369837273160484
            A03 = 0.130371575432137

        if self.SPAMM == 3:
            A = 75.23938332
            B = -11.77882363
            C = 2.75911194
            D = 0.85939469
            E = 0.819414
            F = -3.62893
            G = 5.541968
            H = -2.67478
            J = 0.0000
            K = 0.1735
            L = 4.78043467
            M = 0.13541854
            A00 = 783474.415
            A10 = -18962.5956
            A01 = 21519.0704
            A20 = -117.974195
            A11 = 203.573934
            A02 = -93.0612998
            A30 = -0.0857022781
            A21 = 0.365994533
            A12 = -0.418614264
            A03 = 0.147667014
            B00 = -0.38278304
            B10 = 1.13696746
            B01 = -1.91505933
            B20 = -0.00125747556
            B11 = 0.01133092
            B02 = -0.0249704296
            B30 = 0.00000298600818
            B21 = -0.0000331707251
            B12 = 0.00011459605
            B03 = -0.000119888859
            S = 0.024725
            T = -0.232900
            U = 0.965265
            trap_V = 330

        TTR_from_HAR = E * self.HAR ** 3 + F * self.HAR ** 2 + G * self.HAR + H
        self.energy_eV = trap_V / (A * TTR_from_HAR ** 3 + B * TTR_from_HAR ** 2 + C * TTR_from_HAR + D)
        self.C_E = (A00 + A10 * self.energy_eV + A01 * trap_V + A20 * self.energy_eV ** 2 + A11 * self.energy_eV *
                    trap_V + A02 * trap_V ** 2 + A30 * self.energy_eV ** 3 + A21 * self.energy_eV ** 2 * trap_V +
                    A12 * self.energy_eV * trap_V ** 2 + A03 * trap_V ** 3) ** 2

        self.m_z = self.C_E / self.avg_freq ** 2
        uncorrected_charge = (self.avg_mag + J) / K  # calibration from BSA charge states
        corr_factors = 1 / (L * TTR_from_HAR + M)
        self.charge = uncorrected_charge * corr_factors
        self.mass = self.m_z * self.charge


class Drop:
    def __init__(self, startFreq, endFreq, startMag, endMag, startCharge, endCharge, C_E_initial, C_E_final, t_before,
                 t_after, start_mass, end_mass, fundamental_trace, drop_index, trace_indices, folder):
        # Used to filter out super short drops from analyis
        self.fundamental_trace = fundamental_trace
        self.folder = folder
        self.trace_indices = trace_indices
        self.drop_index = drop_index
        self.drop_index_trace = trace_indices[int(drop_index)]
        self.t_before = t_before
        self.t_after = t_after
        self.startCharge = startCharge
        self.endCharge = endCharge
        self.delta_charge = endCharge - startCharge
        self.start_mass = start_mass
        self.end_mass = end_mass
        self.avg_mass = np.average([start_mass, end_mass])
        # charge_error_guess = np.average([startCharge, endCharge]) * 0.1  # Used to check if charge error can shift
        # the distribution
        self.avg_charge = np.average([startCharge, endCharge])  # + charge_error_guess
        self.freq_change_magnitude = endFreq - startFreq
        self.charge_change_magnitude = endMag - startMag  # Unused in current calculations.
        self.initialFreq = startFreq
        self.finalFreq = endFreq
        self.f_squared_ratio_change = (startFreq ** 2) / (endFreq ** 2)
        self.freq_computed_charge_loss = -(self.f_squared_ratio_change - 1) * self.avg_charge

        # Lets figure out what m/z change this drop accounts for...
        # Check change in energy. If approximately constant, we can say delta_m/z = C_E / delta_f^2
        self.initial_C_E = C_E_initial
        self.final_C_E = C_E_final
        self.delta_C_E = C_E_final - C_E_initial
        self.delta_C_E_percent = (self.delta_C_E / self.initial_C_E) * 100
        # Calculate delta_m/z = C_E / delta_f^2
        self.delta_m_z = self.final_C_E / (self.finalFreq ** 2) - self.final_C_E / (
                self.initialFreq ** 2)  # Observed m/z change
        self.scaled_m_z = (self.delta_m_z / (self.avg_mass / self.avg_charge)) * 100
        self.expected_1C_mz_change = (self.avg_mass / (self.avg_charge - 1)) - (
                self.avg_mass / self.avg_charge)  # The expected m/z change (positive) due to single charge loss
        self.C_loss_scaled_m_z = self.delta_m_z / self.expected_1C_mz_change
        self.delta_mass = self.delta_m_z * self.avg_charge


def gauss(x, A, mu, sigma, offset):
    return offset + A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


if __name__ == "__main__":
    # Uncomment for bulk analysis (point to a folder, then pool all .trace files into one analysis
    ################################################################
    # trace_folders = STFT.choose_top_folders(".traces")
    # print(trace_folders)
    # file_ending = ".trace"
    # filelists = STFT.generate_filelist(trace_folders, file_ending)
    # analysis_name = trace_folders[0].split('.pool', maxsplit=1)[0]
    # new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]
    # analysis_name = analysis_name + '.figures'
    ################################################################

    # Uncomment for single folder analysis (select the .traces folder manually)
    ################################################################
    win = tk.Tk()
    win.focus_force()
    win.withdraw()
    folder = fd.askdirectory(title="Choose traces folder")
    print(folder)
    filelists = STFT.generate_filelist([folder], ".trace")
    file_count = len(filelists[0])
    analysis_name = folder.rsplit('.', maxsplit=1)[0]
    new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]

    ################################################################

    save_plots = True
    # Define font params for exported plots
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    show_plots = True
    smoothed_output = True  # Smooth the histogram before calculating the peak
    f_computed_axv_lines = True
    SPAMM = 3
    drop_threshold = -20  # NOTE: This parameter is affected by the K parameter
    # PLOT SELECTION CONTROLS:
    drops_per_trace = 0
    trace_slope_distribution = 1
    freq_drop_magnitude = 0
    f_computed_charge_loss = 0
    amp_computed_charge_loss = 0
    delta_2D_mass_charge = 0
    C_E_percent_change = 0
    m_z_drop_1D_spectrum = 0
    HAR_eV_distribution = 1
    mass_spectrum_2D = 1
    plot_drop_statistics = 0
    plot_1C_loss_scaled_m_z = 0
    export_demo_drops = False
    demo_trace_range = [0, -50]
    export_demo_traces = True
    # Params for smoothing out the histograms.
    # Currently used for the charge loss plots and the slope distribution plot
    if smoothed_output:
        smooth_polyorder = 2
        smooth_window = 5

    # eV / z boundaries... ions cannot physically exist outside a small range of energies. Set that range here
    ev_z_min = 0  # Default 200
    ev_z_max = 10000  # Default 245

    separation_line_slopes = [0, 15]  # Dividing line between slopes... set to zero for no slope separation
    # Set how many steps the fragments before and after a drop need to exist for to be analyzed here
    # MAJOR RESTRICTION POINT!!!!!!!!!!!!!
    before_existence_threshold = 15
    after_existence_threshold = 15

    max_charge_selection = 1500
    min_charge_selection = 20

    # This value sets the maximum x-axis value that the slope distributions will be plotted with
    slope_max = 0.1
    # Plot all slopes as a background? Disabling this only plots slopes included in analysis.
    plot_comparative_slopes = True
    plot_fit_line = False

    max_mass = 150 * 1000000  # Maximum mass in MDa (only adjust 1st number)
    min_mass = 0 * 1000000  # Minimum mass in MDa (only adjust 1st number)
    max_charge = 1500  # Maximum allowed charges

    # If we only want to look at traces that contain a drop in a specific size range, define that range here
    # Otherwise, set to -20 and +20
    min_drop_search_boundary = -0.7
    max_drop_search_boundary = 1
    # If we want to exclude traces that have erroneously small drops, do that here
    # STRONGLY suggested to use above tool to manually review traces before excluding them
    # TRUE excludes ions in this range, FALSE excludes ions outside this range
    invert_charge_selection = True
    filter_by_drops = False

    # F-Computed charge loss plotting range
    f_computed_plot_min = -4
    f_computed_plot_max = 1
    bin_count = 100  # For the amplitude computed charge loss + the freq-computed charge loss plots

    analysis_name = analysis_name + "_" + str(min_mass / 1000000) + "_to_" + str(max_mass / 1000000) + "MDa"
    # analysis_name = analysis_name + "_" + str(min_charge_selection) + "_to_" + str(max_charge_selection) + "charge"
    # analysis_name = analysis_name + "_" + str(separation_line_slopes[0]) + "_to_" + str(separation_line_slopes[1]) + "slopes"
    analysis_name = analysis_name + '.figures'
    try:
        os.mkdir(analysis_name)
    except FileExistsError:
        print("Path exists already.")
    analysis_name = analysis_name + '/' + new_folder_name

    traces = []
    drops = []
    file_counter = 0
    for file in filelists[0]:
        print("Processing file " + str(file_counter) + " of " + str(file_count))
        newTrace = Trace(file, SPAMM, drop_threshold=drop_threshold)
        file_counter = file_counter + 1
        traces.append(newTrace)
        if len(newTrace.drops) > 0:
            for element in newTrace.drops:
                if element.t_before > before_existence_threshold:
                    if element.t_after > after_existence_threshold:
                        drops.append(element)

    drop_counts = []
    filtered_traces = []
    filtered_drops = []

    mass_collection = []
    charge_collection = []
    m_z_collection = []
    energy_collection = []
    HAR_collection = []
    slope_collection = []
    included_slopes = []
    for trace in traces:
        is_included = True
        fragment_counter = 0
        avg_charge_frags = 0
        avg_mass_frags = 0
        avg_mz_frags = 0
        avg_energy_frags = 0
        avg_HAR_frags = 0
        avg_slope_frags = 0
        include_trace_with_drop = False  # For debugging... try to find out why there is a peak at 0
        for fragment in trace.fragments:
            # Average the mass, charge, HAR, energy, and m/z of all fragments in a trace
            fragment_counter = fragment_counter + 1
            avg_charge_frags = avg_charge_frags + fragment.charge
            avg_mass_frags = avg_mass_frags + fragment.mass
            avg_mz_frags = avg_mz_frags + fragment.m_z
            avg_energy_frags = avg_energy_frags + fragment.energy_eV
            avg_HAR_frags = avg_HAR_frags + fragment.HAR
            avg_slope_frags = avg_slope_frags + (
                    fragment.linfitEquation.coefficients[0] ** 2 / fragment.linfitEquation.coefficients[1] ** 2)

        drop_found_flag = False
        for drop in trace.drops:
            if not drop_found_flag:
                if min_drop_search_boundary < drop.freq_computed_charge_loss < max_drop_search_boundary:
                    drop_found_flag = True
                    if invert_charge_selection:
                        include_trace_with_drop = False
                    else:
                        include_trace_with_drop = True
                else:
                    if invert_charge_selection:
                        include_trace_with_drop = True
                    else:
                        include_trace_with_drop = False
        # Catch case where the ion undergoes zero drops... dont want to exclude then!
        if len(trace.drops) < 1 and not invert_charge_selection:
            include_trace_with_drop = True

        try:
            avg_charge_frags = avg_charge_frags / fragment_counter
            avg_mass_frags = avg_mass_frags / fragment_counter
            avg_mz_frags = avg_mz_frags / fragment_counter
            avg_energy_frags = avg_energy_frags / fragment_counter
            avg_HAR_frags = avg_HAR_frags / fragment_counter
            avg_slope_frags = avg_slope_frags / fragment_counter
            avg_slope_frags = avg_slope_frags * 10.0e7
        except ZeroDivisionError:
            print("ERROR (noncritical): NO fragments in trace / division by zero")

        # If the trace has a slope that is out of bounds, don't include it
        if separation_line_slopes[0] > avg_slope_frags or avg_slope_frags > separation_line_slopes[1]:
            is_included = False
            print('Rejected trace: Slope out of bounds')
        # If the trace has an avg mass that is out of bounds, don't include it either.
        if min_mass > trace.avg_mass or trace.avg_mass > max_mass:
            is_included = False
            print('Rejected trace: Mass out of bounds')
        # If the trace has an avg charge that is out of bounds, don't include it either.
        if min_charge_selection > trace.avg_charge or trace.avg_charge > max_charge_selection:
            is_included = False
            print('Rejected trace: Charge out of bounds')
        # If the energy ev/z is out of range, do not include it
        if ev_z_min > avg_energy_frags or ev_z_max < avg_energy_frags:
            is_included = False
            print('Rejected trace: Energy out of bounds')
        # If the trace contains a 'suspicious' charge loss (use only for debugging)
        if not include_trace_with_drop and filter_by_drops:
            is_included = False
            print('Rejected trace: Drop out of bounds')

        slope_collection.append(avg_slope_frags)
        if is_included:
            filtered_traces.append(trace)
            mass_collection.append(avg_mass_frags)
            charge_collection.append(avg_charge_frags)
            m_z_collection.append(avg_mz_frags)
            energy_collection.append(avg_energy_frags)
            HAR_collection.append(avg_HAR_frags)
            included_slopes.append(avg_slope_frags)
            drop_counts.append(len(trace.drops))
            if len(trace.drops) > 0:
                for element in trace.drops:
                    if element.t_before > before_existence_threshold:
                        if element.t_after > after_existence_threshold:
                            filtered_drops.append(element)

    traces = filtered_traces
    drops = filtered_drops
    figure_counter = 0
    for trace in traces:
        if export_demo_traces:
            try:
                print("Exporting TRX " + str(figure_counter) + " of " + str(len(traces)))
                background = ZxxToolkit.ZxxBackdrop(trace.folder)
                bg_save_folder = str(analysis_name)
                try:
                    os.mkdir(bg_save_folder)
                except FileExistsError:
                    if False:  # Got sick of reading this warning.....
                        print("TrX base path exists already.")
                bg_save_folder = bg_save_folder + '/TrX'
                try:
                    os.mkdir(bg_save_folder)
                except FileExistsError:
                    if False:  # Got sick of reading this warning.....
                        print("TrX path exists already.")
                background.plot_zoomed_trace_on_Zxx(trace, str(figure_counter), bg_save_folder, save_plots=True)
                figure_counter = figure_counter + 1
            except Exception as e:
                print("Error exporting demo traces: ", e)

    dropsSquaredRatioChange = []
    dropsMagnitude = []
    dropsCharge = []
    dropsChargeChange = []  # Amplitude computed
    freqComputedChargeLoss = []  # Frequency computed
    delta_m_z = []
    C_loss_scaled_m_z = []
    scaled_delta_m_z = []
    delta_C_E = []
    delta_C_E_percent = []
    delta_mass = []
    delta_charge = []  # Frequency computed here, amplitude computed in drop objects
    figure_counter = 0
    for drop in drops:
        dropsMagnitude.append(float(drop.freq_change_magnitude))
        dropsCharge.append(float(drop.charge_change_magnitude))
        dropsSquaredRatioChange.append(float(drop.f_squared_ratio_change))
        C_loss_scaled_m_z.append(drop.C_loss_scaled_m_z)
        delta_m_z.append(drop.delta_m_z)
        delta_C_E.append(drop.delta_C_E)
        delta_C_E_percent.append(drop.delta_C_E_percent)
        scaled_delta_m_z.append(drop.scaled_m_z)
        delta_mass.append(drop.delta_mass)
        delta_charge.append(drop.freq_computed_charge_loss)
        if 30 > drop.delta_charge > -30:
            dropsChargeChange.append(float(drop.delta_charge))
        if 30 > drop.freq_computed_charge_loss > -30:
            freqComputedChargeLoss.append(float(drop.freq_computed_charge_loss))
            if demo_trace_range[0] > drop.freq_computed_charge_loss > demo_trace_range[1] and export_demo_drops:
                try:
                    print("Exporting DRX " + str(figure_counter) + " of " + str(len(drops)))
                    background = ZxxToolkit.ZxxBackdrop(drop.folder)
                    bg_save_folder = str(analysis_name)
                    try:
                        os.mkdir(bg_save_folder)
                    except FileExistsError:
                        print("DrX base path exists already.")
                    bg_save_folder = bg_save_folder + '/DrX'
                    try:
                        os.mkdir(bg_save_folder)
                    except FileExistsError:
                        print("DrX path exists already.")
                    background.plot_zoomed_drop_on_Zxx(drop, drop.drop_index_trace, str(figure_counter), bg_save_folder,
                                                       save_plots=True)
                    figure_counter = figure_counter + 1
                except Exception as e:
                    print("Error exporting demo drop: ", e)

    if drops_per_trace:
        plt.hist(drop_counts, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plt.title("Drops/Trace")
        plt.xlabel('Drops')
        plt.ylabel('Counts')
        if save_plots:
            plt.savefig(str(analysis_name) + '_drops_per_trace.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if trace_slope_distribution:
        # Only is plotting slopes THAT ARE COUNTED IN THE MASS SPECTRUM
        if plot_comparative_slopes:
            plt.hist(slope_collection, bins=150, range=[0, slope_max])
        hist_out = plt.hist(included_slopes, color="red", bins=150, range=[0, slope_max])
        if smoothed_output:
            # plt.clf()
            bins = hist_out[1][0:-1]
            counts = sig.savgol_filter(hist_out[0], window_length=smooth_window, polyorder=smooth_polyorder)
            # plt.plot(bins, counts, linestyle="solid", color="black")
        else:
            bins = hist_out[1][0:-1]
            counts = hist_out[0]

        peak_indices, properties = sig.find_peaks(counts, width=2, distance=bin_count * 0.05,
                                                  prominence=max(counts) * 0.25)

        for peak in bins[peak_indices]:
            dx = bin_count * 0.01
            A_constraints = [-np.inf, np.inf]  # Amplitude
            mu_constraints = [peak - dx, peak + dx]  # x-axis flexibility
            sigma_constraints = [0, bin_count * 0.1]  # Peak width
            offset_constraints = [0, 0.1]  # Vertical offset
            lower_bounds = [A_constraints[0], mu_constraints[0], sigma_constraints[0], offset_constraints[0]]
            upper_bounds = [A_constraints[1], mu_constraints[1], sigma_constraints[1], offset_constraints[1]]
            try:
                param, param_cov = curve_fit(gauss, bins, np.array(counts), bounds=(lower_bounds, upper_bounds))
                peak_contrib_to_slice = gauss(bins, param[0], param[1], param[2], param[3])
                if plot_fit_line:
                    plt.plot(bins, peak_contrib_to_slice, linestyle="dashdot")
                    text_string = f'{param[1]:.2f}'
                    plt.text(param[1] + 1, param[0] + param[0] * 0.05, text_string)
            except Exception:
                print("Could not fit peak at ", str(peak))

        plt.title("Trace Slope Distribution")
        plt.xlabel('Slope: Drift (Hz) per STFT Step (5ms)')
        plt.ylabel('Counts')
        if save_plots:
            plt.savefig(str(analysis_name) + '_slope_distribution.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if plot_drop_statistics:
        # Plot drop count vs starting frequency
        starting_freqs_no_drop = []
        starting_freqs_one_drop = []
        starting_freqs_two_drops = []
        starting_freqs_three_drops = []
        for trace in traces:
            if len(trace.drops) == 1:
                starting_freqs_one_drop.append(trace.fragments[0].avg_freq)
            elif len(trace.drops) == 2:
                starting_freqs_two_drops.append(trace.fragments[0].avg_freq)
            elif len(trace.drops) == 3:
                starting_freqs_three_drops.append(trace.fragments[0].avg_freq)
            elif len(trace.drops) == 0:
                starting_freqs_no_drop.append(np.average(trace.trace))
        plt.subplot(1, 3, 1)
        plt.hist(starting_freqs_no_drop, 10)
        plt.title("Starting Frequency (0 drop)")
        plt.xlabel('Frequency')
        plt.ylabel('Counts')
        plt.subplot(1, 3, 2)
        plt.hist(starting_freqs_one_drop, 10)
        plt.title("Starting Frequency (1 drop)")
        plt.xlabel('Frequency')
        plt.ylabel('Counts')
        plt.subplot(1, 3, 3)
        plt.hist(starting_freqs_two_drops, 10)
        plt.title("Starting Frequency (2 drop)")
        plt.xlabel('Frequency')
        plt.ylabel('Counts')
        if save_plots:
            plt.savefig(str(analysis_name) + '_drop_statistics.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    # Plot drop freq magnitude
    freq_drop_magnitudes = []
    for drop in drops:
        freq_drop_magnitudes.append(drop.freq_change_magnitude)

    if freq_drop_magnitude:
        plt.hist(freq_drop_magnitudes, 100, range=[-400, 0])
        plt.title("Freq Drop Magnitude")
        plt.xlabel('Frequency')
        plt.ylabel('Counts')
        if save_plots:
            plt.savefig(str(analysis_name) + '_freq_drop_magnitude.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if f_computed_charge_loss:
        # Plot frequency computed charge loss alongside directly computed charge loss
        hist_out = plt.hist(dropsChargeChange, bin_count, range=[-20, 15])
        if smoothed_output:
            # plt.clf()
            bins = hist_out[1][0:-1]
            counts = sig.savgol_filter(hist_out[0], window_length=smooth_window, polyorder=smooth_polyorder)
            # plt.plot(bins, counts, linestyle="solid", color="black")
        else:
            bins = hist_out[1][0:-1]
            counts = hist_out[0]

        A_constraints = [-np.inf, np.inf]
        mu_constraints = [-np.inf, np.inf]
        sigma_constraints = [0, np.inf]
        offset_constraints = [0, 1]
        lower_bounds = [A_constraints[0], mu_constraints[0], sigma_constraints[0], offset_constraints[0]]
        upper_bounds = [A_constraints[1], mu_constraints[1], sigma_constraints[1], offset_constraints[1]]
        param, param_cov = curve_fit(gauss, bins, np.array(counts), bounds=(lower_bounds, upper_bounds))

        peak_contrib_to_slice = gauss(bins, param[0], param[1], param[2], param[3])

        if amp_computed_charge_loss:
            plt.plot(bins, peak_contrib_to_slice, linestyle="dashdot")
            text_string = f'{param[1]:.2f}'
            plt.text(param[1] + 5, param[0] - 1.5, text_string, fontsize=16)
            print("Amplitude Computed Peak Center: ", str(param[1]))

            plt.title("Amplitude Only Charge Loss")
            plt.xlabel('Charge')
            plt.ylabel('Counts')
            plt.xticks([-20, -16, -12, -8, -4, 0, 4, 8, 12])
            if save_plots:
                plt.savefig(str(analysis_name) + '_amp_computed_charge_loss.png')
            if show_plots:
                plt.show()
            else:
                plt.close()

        hist_out = plt.hist(freqComputedChargeLoss, bin_count, range=[f_computed_plot_min, f_computed_plot_max])
        if smoothed_output:
            # plt.clf()
            bins = hist_out[1][0:-1]
            counts = sig.savgol_filter(hist_out[0], window_length=smooth_window, polyorder=smooth_polyorder)
            # plt.plot(bins, counts, linestyle="solid", color="black")
        else:
            bins = hist_out[1][0:-1]
            counts = hist_out[0]

        # Comment out for SINGLE gaussian fitting
        bin_spacing = abs(bins[1] - bins[0])
        peak_indices, properties = sig.find_peaks(counts, width=2, distance=bin_count * 0.05,
                                                  prominence=max(counts) * 0.25)
        peak_counter = 0

        if f_computed_axv_lines:
            for peak in bins[peak_indices]:
                plt.axvline(peak, color='orange', linestyle='dashdot')
                text_string = f'{peak:.2f}'
                plt.text(peak - 4, max(counts), text_string)

        #     dx = bin_count * 0.0002
        #     A_constraints = [-np.inf, np.inf]  # Amplitude
        #     mu_constraints = [peak - dx, peak + dx]  # x-axis flexibility
        #     sigma_constraints = [0, bin_count * 0.1]  # Peak width
        #     offset_constraints = [0, 0.1]  # Vertical offset
        #     lower_bounds = [A_constraints[0], mu_constraints[0], sigma_constraints[0], offset_constraints[0]]
        #     upper_bounds = [A_constraints[1], mu_constraints[1], sigma_constraints[1], offset_constraints[1]]
        #     try:
        #         param, param_cov = curve_fit(gauss, bins, np.array(counts), bounds=(lower_bounds, upper_bounds))
        #
        #         if -5 < param[1] < 0:
        #             peak_contrib_to_slice = gauss(bins, param[0], param[1], param[2], param[3])
        #             plt.plot(bins, peak_contrib_to_slice, linestyle="dashdot")
        #             text_string = f'{param[1]:.2f}'
        #             plt.text(param[1] - 0.5, param[0] - 1.5, text_string)
        #             print("Freq Computed Peak Center: ", str(param[1]))
        #     except Exception:
        #         print("Could not fit peak at ", str(peak))
        #
        #     peak_counter = peak_counter + 1

        plt.title("Frequency Computed Charge Loss")
        plt.xlabel('Charge')
        plt.ylabel('Counts')
        plt.xticks(range(f_computed_plot_min, f_computed_plot_max))
        if save_plots:
            plt.savefig(str(analysis_name) + '_f_computed_charge_loss.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if plot_1C_loss_scaled_m_z:
        plt.hist(C_loss_scaled_m_z, bins=200, range=[0, 5])
        plt.title("C_loss_scaled_m_z")
        plt.xlabel("(delta m/z) / (expected 1C delta m/z)")
        if save_plots:
            plt.savefig(str(analysis_name) + '_1c_loss_scaled_mz_change.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if delta_2D_mass_charge:
        # Plot scatter of delta m vs delta charge
        # plt.subplot(1, 2, 1)  # row 1, col 2 index 1
        # plt.hist2d(scaled_delta_m_z, delta_charge, bins=100, range=[[0, 4], [-5, 0]], cmap='nipy_spectral_r')
        # plt.colorbar()
        # plt.title("Delta m/z")
        # plt.xlabel('m/z Change (%)')
        # plt.ylabel('Charge Change')

        avg_mass = np.average(mass_collection)
        std_mass = np.std(mass_collection)
        avg_charge = np.average(charge_collection)
        std_charge = np.std(charge_collection)

        heatmap, xedges, yedges = np.histogram2d(scaled_delta_m_z, delta_charge, bins=[160, 120],
                                                 range=[[0, 5], [-4, -0.35]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        gaussmap = gaussian_filter(heatmap, 1, mode='nearest')

        # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
        plt.imshow(gaussmap.T, cmap='nipy_spectral_r', extent=extent, origin='lower', aspect='auto',
                   interpolation='none')

        mass_line_1 = avg_mass  # Reference line for charge loss
        charge_line_1 = avg_charge
        y_points = np.linspace(-4, -0.35, 100)
        x = 100 * (1 - (mass_line_1 / (charge_line_1 - y_points)) / (mass_line_1 / charge_line_1))
        plt.plot(x, y_points, linestyle="dashdot")

        mass_line_2 = avg_mass  # Reference line for charge loss
        charge_line_2 = avg_charge + std_charge
        y_points = np.linspace(-4, -0.35, 100)
        x = 100 * (1 - (mass_line_2 / (charge_line_2 - y_points)) / (mass_line_2 / charge_line_2))
        plt.plot(x, y_points, linestyle="solid", color="black")

        mass_line_3 = avg_mass  # Reference line for charge loss
        charge_line_3 = avg_charge - std_charge
        y_points = np.linspace(-4, -0.35, 100)
        x = 100 * (1 - (mass_line_3 / (charge_line_3 - y_points)) / (mass_line_3 / charge_line_3))
        plt.plot(x, y_points, linestyle="solid", color="blue")

        text_string = f'Avg Charge: {avg_charge:.2f} +/- {std_charge:.2f}'
        plt.text(1.5, -0.5, text_string)
        text_string = f'Avg Mass: {avg_mass:.2f} +/- {std_mass:.2f}'
        plt.text(1.5, -0.8, text_string)

        plt.colorbar()
        plt.title("Delta m/z (smoothed)")
        plt.xlabel('m/z Change (%)')
        plt.ylabel('Charge Change')
        if save_plots:
            plt.savefig(str(analysis_name) + '_delta_2D_mass_charge.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if C_E_percent_change:
        # Plot percent CE change
        plt.hist(delta_C_E_percent, 100, range=[0, 100])
        plt.title("Relative Percent Change in CE")
        plt.xlabel('Relative Percent')
        plt.ylabel('Counts')
        if save_plots:
            plt.savefig(str(analysis_name) + '_C_E_percent_change.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if m_z_drop_1D_spectrum:
        # Plot change in m/z
        plt.hist(delta_m_z, 150, range=[-200, 1500])
        plt.title("m/z Change Associated with Drops")
        plt.xlabel('delta m/z')
        plt.ylabel('Counts')
        if save_plots:
            plt.savefig(str(analysis_name) + '_m_z_drop_1D_spectrum.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if HAR_eV_distribution:
        # Plot HAR distribution and energy (eV) distribution
        plt.subplot(1, 2, 1)  # row 1, col 2 index 1
        plt.hist(HAR_collection, 100)
        plt.title("HAR Distribution")
        plt.xlabel('HAR')
        plt.ylabel('Counts')

        plt.subplot(1, 2, 2)  # index 2
        plt.hist(energy_collection, 100)
        plt.title("Ion Energy (eV) Distribution")
        plt.xlabel('Energy (eV)')
        plt.ylabel('Counts')
        if save_plots:
            plt.savefig(str(analysis_name) + '_HAR_eV_distribution.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

    if mass_spectrum_2D:
        # Plot a 2D mass spectrum
        plot_rayleigh_line()
        heatmap, xedges, yedges = np.histogram2d(mass_collection, charge_collection, bins=[160, 120],
                                                 range=[[0, 150000000], [0, 1500]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        gaussmap = gaussian_filter(heatmap, 1, mode='nearest')

        # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
        plt.imshow(gaussmap.T, cmap='nipy_spectral_r', extent=extent, origin='lower', aspect='auto',
                   interpolation='none')

        # plt.hist2d(mass_collection, charge_collection, bins=125, range=[[500000, 20000000], [0, 500]], cmap='nipy_spectral_r')
        plt.colorbar()
        plt.title("2D Mass Spectrum")
        plt.xlabel('Mass (MDa)')
        plt.ylabel('Charge')
        # x = np.multiply(np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]), 1000000)
        x = np.multiply(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]), 1000000)
        # labels = ["0", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20"]
        labels = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "150"]
        plt.xticks(x, labels)
        if save_plots:
            plt.savefig(str(analysis_name) + '_mass_spectrum_2D.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
try:
    print("Selected data includes " + str(len(traces)) + " valid ions and " + str(
        len(drops)) + " recorded emission events.")

except Exception:
    print('Called by import...')
