'''
performs a bootstrap analysis on egocentric head-centered data, calculating metrics such as mean resultant length (MRL), mean angle, and preferred firing distance for neural activity based on head orientation. It processes spike data, computes distance and angle bins, fits a Weibull distribution to determine the preferred distance, and saves MRL thresholds for further statistical validation.
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

from utils import set_to_nan_based_on_likelihood, plot_ebc,filter_and_interpolate
from egocentric_head import *
from Egocentric import *
from scipy.stats import weibull_min

def calculate_mr(firing_rates, n, m):
    """
    Calculate the mean resultant vector (MR) for given firing rates.

    Args:
        firing_rates (ndarray): Array of firing rate data.
        n (int): Number of orientation bins.
        m (int): Number of distance bins.

    Returns:
        complex: Mean resultant vector (MR), capturing direction and strength of orientation.
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    MR = (1 / (n * m)) * np.sum(firing_rates * np.exp(1j * theta[:, None]), axis=(0, 1))
    return MR

def shuffle_spike_train(spike_times, recording_duration, min_shift=30):
    """
    Shuffle spike train data by randomly shifting spike times within the recording duration.

    Args:
        spike_times (array-like): Original spike times to be shuffled.
        recording_duration (float): Total recording duration.
        min_shift (int): Minimum allowable shift interval (default is 30).

    Returns:
        ndarray: Shuffled spike times, shifted by a random interval.
    """
    # Determine maximum shift based on recording duration
    max_shift = recording_duration - min_shift
    
    # Generate a random shift interval
    shift = np.random.uniform(min_shift, max_shift)
    
    # Shift the spike times
    shuffled_times = (spike_times + shift) % recording_duration
    
    return shuffled_times

def bootstrap_egocentric_head(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, file,speed_threshold, ebc_angle_bin_size,ebc_dist_bin_size, dist_bins):
    """
    Perform bootstrap analysis on egocentric head-centered EBC data and calculate various metrics.

    Args:
        dlc_df (DataFrame): Dataframe containing body coordinate and timestamp data.
        phy_df (DataFrame): Dataframe with spike time data for each cell.
        fps (int): Frames per second of video recording.
        likelihood_threshold (float): Threshold to filter out low-likelihood data.
        model_dt (float): Time step for interpolating model data.
        bin_width (int): Width of bins for analysis.
        file (str): Filename for saving results.
        speed_threshold (float): Minimum speed threshold for filtering.
        ebc_angle_bin_size (int): Size of angle bins in degrees.
        ebc_dist_bin_size (int): Size of distance bins.

    Returns:
        tuple: Contains MRLs, MRL thresholds, mean angles, EBC plot data, distance bins, binary plot data,
               max bins, and preferred distances for further analysis.
    """
    columns_of_interest = ['driveL','driveR','time']

    # Adding timestamps to dlc file and only considering columns of interest
    dlc_df['time'] = np.arange(len(dlc_df))/fps

    #filter and interpolate
    model_data_df,model_t = filter_and_interpolate(dlc_df,columns_of_interest,likelihood_threshold,model_dt,fps)

    model_data_df = model_data_df[model_data_df['speed']>speed_threshold]

    #model_data_df = model_data_df.dropna()

    center_neck_x = list(model_data_df['driveL x'])
    center_neck_y = list(model_data_df['driveL y'])
    center_haunch_x = list(model_data_df['driveR x'])
    center_haunch_y = list(model_data_df['driveR y'])

    egocentric_file = file[:-3]+'ebc_head_data'
    ebc_data = calaculate_ebc_head(dlc_df, center_neck_x,center_neck_y,center_haunch_x,center_haunch_y,ebc_angle_bin_size,ebc_dist_bin_size)
    
    distance_bins,angle_bins = ebc_bins(dlc_df,ebc_angle_bin_size,ebc_dist_bin_size)

    ebc_data_avg = np.sum(ebc_data,axis=0)
    ebc_data_avg = ebc_data_avg[:dist_bins, :]
    
    model_data_df['egocentric'] = list(ebc_data)

    distance_bins = distance_bins[:dist_bins] #cut off far half of the arena
    rbins = distance_bins.copy()
    abins = np.linspace(0,2*np.pi, (360//ebc_angle_bin_size))

    recording_duration = len(model_data_df)/fps

    cell_numbers = phy_df.index
    ebc_plot_data = []

    ebc_plot_data_binary = []   
    max_bins = []

    MRLS = []
    MALS = []
    n = int(360 // ebc_angle_bin_size)  # number of orientation bins
    m = dist_bins   # number of distance bins
    n_bootstrap = 100  # number of bootstrap iterations
    mrl_thresholds = []
    preferred_dist = []

    for i in cell_numbers:
        spike_times = phy_df.loc[i]['spikeT']

        #removing spike times after camera stopped
        spike_times = spike_times[spike_times<=max(model_t)]

        shuffled_mrls = []

        #binning spikes
        sp_count_ind = np.digitize(spike_times,bins = model_t)
        
        #-1 because np.digitze is 1-indexed
        sp_count_ind = [i-1 for i in sp_count_ind]

        sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]
        
        cell_spikes_egocentric = model_data_df['egocentric'].loc[sp_count_ind]  
        
        cell_spikes_avg = np.sum(cell_spikes_egocentric,axis = 0)
        
        cell_spikes_avg = np.divide(cell_spikes_avg,ebc_data_avg)
        
        cell_spikes_avg[np.isnan(cell_spikes_avg)] = 0

        #"half the arena size" filter
        cell_spikes_avg = cell_spikes_avg[:dist_bins]

        ebc_plot_data.append(cell_spikes_avg)

        firing_rates = cell_spikes_avg.copy().T

        theta = abins.copy()
        MR = (1 / (n * m)) * np.sum(firing_rates * np.exp(1j * theta[:, None]), axis=(0, 1))
        MRL = np.abs(MR)
        MRA = np.angle(MR)

        MRLS.append(MRL)
        MALS.append(MRA)
        
        
        #75% threshold    
        max_idx = np.unravel_index(np.argmax(cell_spikes_avg[:24], axis=None), cell_spikes_avg[:24].shape)

        # Corresponding radius and angle for the maximum value
        max_radius = rbins[max_idx[0]]
        max_angle = abins[max_idx[1]]

        threshold = np.percentile(cell_spikes_avg, 75)

        # Convert the array into a binary array
        binary_array = np.where(cell_spikes_avg >= threshold, 1, 0)

        ebc_plot_data_binary.append(binary_array)

        max_bins.append([max_angle,max_radius])

        # Extract the firing rate vector along the preferred angle MRA
        # Here we're assuming you calculate this by taking the firing rates at the closest orientation to MRA
        preferred_orientation_idx = np.argmin(np.abs(theta - MRA))
        firing_rate_vector = firing_rates[preferred_orientation_idx, :]

        # Fit a Weibull distribution
        params = weibull_min.fit(firing_rate_vector)

        # Get the distance bin with the maximum estimated firing rate
        max_firing_distance_bin = np.argmax(weibull_min.pdf(np.arange(m), *params))

        preferred_dist.append(max_firing_distance_bin)


        bootstrap_file = file[:-3]+'_bootstrap_ebc_head_data'
        for _ in range(n_bootstrap):

            shuffled_spikes = shuffle_spike_train(spike_times, recording_duration)

            #binning spikes
            sp_count_ind = np.digitize(shuffled_spikes,bins = model_t)

            #-1 because np.digitze is 1-indexed
            sp_count_ind = [i-1 for i in sp_count_ind]

            sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]

            cell_spikes_egocentric = model_data_df['egocentric'].loc[sp_count_ind]  

            cell_spikes_avg = np.sum(cell_spikes_egocentric,axis = 0)
            cell_spikes_avg = np.divide(cell_spikes_avg,ebc_data_avg)
            
            cell_spikes_avg[np.isnan(cell_spikes_avg)] = 0
            

            shuffled_firing_rates = cell_spikes_avg.copy().T
            theta = abins.copy()
            MR = (1 / (n * m)) * np.sum(shuffled_firing_rates * np.exp(1j * theta[:, None]), axis=(0, 1))
            MRL = np.abs(MR)

            # Append to the shuffled MRLs list
            shuffled_mrls.append(MRL)
        mrl_threshold = np.percentile(shuffled_mrls, 99)
        mrl_thresholds.append(mrl_threshold)

    return MRLS,mrl_thresholds,MALS,ebc_plot_data, distance_bins,ebc_plot_data_binary, max_bins,preferred_dist