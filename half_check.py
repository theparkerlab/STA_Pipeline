'''
conducts a half-check analysis on egocentric head-centered data by splitting the dataset into two halves. Each half is processed to filter, interpolate, and calculate spatially tuned metrics, such as mean resultant lengths (MRLs), mean angles, and preferred distances for each cell's spike data. The results are saved and compared to assess consistency in head-centered egocentric firing patterns across the two data subsets.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import cv2

from utils import set_to_nan_based_on_likelihood, plot_ebc, filter_and_interpolate
from egocentric_head import *
from Egocentric import *
# from scipy.stats import weibull_min
from scipy.optimize import curve_fit

from half_split import dlc_interleaved_masks

def _weibull_with_baseline(x, c, a, k, lam):
    """Baseline + scaled Weibull so fit can be high at x=0 (Weibull PDF is zero at origin)."""
    x_safe = np.maximum(x.astype(float), 1e-6)
    w = (k / lam) * ((x_safe / lam) ** (k - 1)) * np.exp(-((x_safe / lam) ** k))
    return c + a * w


def _fit_weibull(x, y):
    """Fit baseline + scaled Weibull to firing rate vs distance. Returns (fit_success, y_fit) or (False, None)."""
    try:
        max_y = np.nanmax(y)
        min_y = np.nanmin(y)
        if max_y <= 0 or len(x) < 4:
            return False, None
        # Initial baseline ~ value at first bin; peak amplitude; shape/scale
        p0 = (float(y.flat[0]) if y.size else 0, max_y - min_y, 2.0, np.median(x[x > 0]) if np.any(x > 0) else 50.0)
        bounds = ([0, 0, 0.5, 1e-6], [max_y * 2, max_y * 500, 10, np.max(x) * 2])
        popt, _ = curve_fit(_weibull_with_baseline, x, y, p0=p0, bounds=bounds, maxfev=3000) # this is implemented differently in the rest of the code, might want to unify PRLP 03/20/2026
        y_fit = _weibull_with_baseline(x, *popt)
        return True, y_fit
    except Exception:
        return False, None

def process_half(dlc_df_half, columns_of_interest, likelihood_threshold, model_dt, fps, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size, dist_bins):
    """
    Processes one half of the dlc DataFrame to filter, interpolate, and calculate egocentric head-centered (EBC) data.

    Args:
        dlc_df_half (DataFrame): A half of the full dlc DataFrame to process.
        columns_of_interest (list): List of columns to retain in the filtered data.
        likelihood_threshold (float): Minimum likelihood threshold for data filtering.
        model_dt (float): Time step for interpolating data.
        fps (int): Frames per second of video recording.
        speed_threshold (float): Minimum speed threshold for filtering.
        ebc_angle_bin_size (int): Size of angle bins in degrees.
        ebc_dist_bin_size (int): Size of distance bins.

    Returns:
        tuple: Processed model DataFrame with egocentric data, and the model time array.
    """
    

    # Filter and interpolate
    model_data_df, model_t = filter_and_interpolate(dlc_df_half, columns_of_interest, likelihood_threshold, model_dt, fps)
    
    # Filter based on speed threshold
    model_data_df = model_data_df[model_data_df['speed'] > speed_threshold]
    
    # Get coordinates for ebc calculation
    center_neck_x = list(model_data_df['driveL x'])
    center_neck_y = list(model_data_df['driveL y'])
    center_haunch_x = list(model_data_df['driveR x'])
    center_haunch_y = list(model_data_df['driveR y'])
    
    # Calculate ebc
    ebc_data = calaculate_ebc_head(dlc_df_half, center_neck_x, center_neck_y, center_haunch_x, center_haunch_y, ebc_angle_bin_size, ebc_dist_bin_size)
    distance_bins, angle_bins = ebc_bins(dlc_df_half, ebc_angle_bin_size, ebc_dist_bin_size)
    
    ebc_data_avg = np.sum(ebc_data,axis=0)
    ebc_data_avg = ebc_data_avg[:dist_bins, :]
    distance_bins = distance_bins[:dist_bins]
    rbins = distance_bins.copy()
    abins = np.linspace(0, 2*np.pi, 121)
    
    model_data_df['egocentric'] = list(ebc_data)
    
    return model_data_df, model_t

def calc_mrls(model_data_df, phy_df, cell_numbers, model_t, abins, ebc_angle_bin_size, dist_bins,fps):
    """
    Calculate mean resultant lengths (MRLs) and preferred distances for each cell's spike data.

    Args:
        model_data_df (DataFrame): Processed model data with egocentric information.
        phy_df (DataFrame): DataFrame with spike time data for each cell.
        cell_numbers (Index): List of cell identifiers.
        model_t (ndarray): Array of time indices for the model data.
        abins (ndarray): Angle bins for calculating preferred angles.

    Returns:
        tuple: MRLs, mean angles, and preferred distances for each cell.
    """
    filt_size = 3 # size of bins for gaussian filter
    
    ebc_plot_data = []
    n = int(360 // ebc_angle_bin_size)  # number of orientation bins
    m = dist_bins   # number of distance bins
    MRLS = []
    MALS = []
    preferred_dist = []
    for i in cell_numbers:
        spike_times = phy_df.loc[i]['spikeT']

        # Removing spike times after camera stopped
        spike_times = spike_times[spike_times <= max(model_t)]

        # Binning spikes
        sp_count_ind = np.digitize(spike_times, bins=model_t)

        # -1 because np.digitize is 1-indexed
        sp_count_ind = [i-1 for i in sp_count_ind]

        sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]

        cell_spikes_egocentric = model_data_df['egocentric'].loc[sp_count_ind]  

        cell_spikes_avg = np.sum(cell_spikes_egocentric, axis=0)
        
        #"occupancy" data
        ebc_data_avg = np.sum(np.array(model_data_df['egocentric']), axis=0)
        ebc_data_avg = ebc_data_avg[:dist_bins, :]

        #"half the arena size" filter
        cell_spikes_avg = cell_spikes_avg[:dist_bins,:]

        # Normalize by occupancy (match bootstrap). Use NaN-protection to avoid divide-by-zero.
        cell_spikes_avg = np.divide(cell_spikes_avg, ebc_data_avg)
        
        cell_spikes_avg[np.isnan(cell_spikes_avg)] = 0
        cell_spikes_avg[np.isinf(cell_spikes_avg)] = 0

        cell_spikes_avg = np.multiply(cell_spikes_avg, fps)

        cell_spikes_avg = cv2.GaussianBlur(cell_spikes_avg,(filt_size,filt_size),filt_size)
        
        ebc_plot_data.append(cell_spikes_avg)

        firing_rates = cell_spikes_avg.copy().T
        mean_firing_rate = np.mean(firing_rates)
        theta = abins.copy()
        MR = (1 / (n * m)) * np.sum(firing_rates * np.exp(1j * theta[:, None]), axis=(0, 1))
        MR = MR / mean_firing_rate if mean_firing_rate != 0 else 0
        MRL = np.abs(MR)
        MRA = np.angle(MR)

        MRLS.append(MRL)
        MALS.append(MRA)
        preferred_orientation_idx = np.argmin(np.abs(theta - MRA))
        firing_rate_vector = firing_rates[preferred_orientation_idx, :]
        # max_firing_distance_bin = np.argmax(firing_rate_vector)

        # Fit a Weibull distribution
        fit_ok, y_fit = _fit_weibull(np.arange(len(firing_rate_vector)), firing_rate_vector)
        peak_idx = np.argmax(y_fit) if fit_ok else np.argmax(firing_rate_vector)
        preferred_dist.append(peak_idx)

        #old weibull fit (not working well, turned off PRLP 03/27/2026)
        # params = weibull_min.fit(firing_rate_vector)
        # # Get the distance bin with the maximum estimated firing rate
        # max_firing_distance_bin = np.argmax(weibull_min.pdf(np.arange(m), *params)) #not fitting correctly, turned off PRLP 7/28/25
        
    return MRLS, MALS, preferred_dist, ebc_plot_data


def egocentric_head_half_check(
    dlc_df,
    phy_df,
    fps,
    likelihood_threshold,
    model_dt,
    bin_width,
    file,
    speed_threshold,
    ebc_angle_bin_size,
    ebc_dist_bin_size,
    dist_bins,
    half_split_mode="interleaved",
    interleave_block_sec=10.0,
):
    """
    Perform a half-check analysis on egocentric head-centered data by splitting the data and analyzing each half separately.

    Args:
        dlc_df (DataFrame): DataFrame containing coordinates and timestamps for the body.
        phy_df (DataFrame): DataFrame with spike time data for each cell.
        fps (int): Frames per second of video recording.
        likelihood_threshold (float): Threshold to filter out low-likelihood data.
        model_dt (float): Time step for interpolating model data.
        bin_width (int): Width of bins for analysis.
        file (str): Filename for saving results.
        speed_threshold (float): Minimum speed threshold for filtering.
        ebc_angle_bin_size (int): Size of angle bins in degrees.
        ebc_dist_bin_size (int): Size of distance bins.

    Returns:
        tuple: MRLs, mean angles, and preferred distances for each half of the data, allowing comparison.
    """

    columns_of_interest = ['driveL', 'driveR', 'time']

    # Adding timestamps to dlc file and only considering columns of interest
    dlc_df['time'] = np.arange(len(dlc_df)) / fps #is this working right? is the dt=1/fps? default for arange is 1...

    # Split dlc_df into two pseudo-halves (contiguous or time-interleaved blocks).
    n_frames = len(dlc_df)
    times_s = np.arange(n_frames) / fps
    mask_a, mask_b = dlc_interleaved_masks(times_s, n_frames, half_split_mode, interleave_block_sec)
    dlc_df_1 = dlc_df.iloc[mask_a].copy()
    dlc_df_2 = dlc_df.iloc[mask_b].copy()

    # Process each half
    model_data_df_1, model_t1 = process_half(dlc_df_1, columns_of_interest, likelihood_threshold, model_dt, fps, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size, dist_bins)
    model_data_df_2, model_t2 = process_half(dlc_df_2, columns_of_interest, likelihood_threshold, model_dt, fps, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size, dist_bins)

    cell_numbers = phy_df.index
    distance_bins, angle_bins = ebc_bins(dlc_df, ebc_angle_bin_size, ebc_dist_bin_size)
    distance_bins = distance_bins[:dist_bins]
    rbins = distance_bins.copy()
    abins = np.linspace(0, 2*np.pi, (360 // ebc_angle_bin_size))

    half_check_file = file[:-3]+'_half_ebc_head_data'
    MRLS_1, MALS_1, pref_dist_1, ebc_plot_data_1 = calc_mrls(model_data_df_1, phy_df, cell_numbers, model_t1, abins, ebc_angle_bin_size, dist_bins, fps)
    MRLS_2, MALS_2, pref_dist_2, ebc_plot_data_2 = calc_mrls(model_data_df_2, phy_df, cell_numbers, model_t2, abins, ebc_angle_bin_size, dist_bins, fps)

    return MRLS_1, MRLS_2, MALS_1, MALS_2, pref_dist_1, pref_dist_2, ebc_plot_data_1, ebc_plot_data_2
