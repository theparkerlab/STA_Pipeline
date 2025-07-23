'''
performs a bootstrap analysis on egocentric body-centered data to calculate metrics such as mean resultant length (MRL) and preferred distance for firing rates. It processes spike data, calculates orientation and distance bins, and fits a Weibull distribution to identify preferred distances. The results, including thresholded MRL values, are saved for further analysis.
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
from tqdm import tqdm

from utils import set_to_nan_based_on_likelihood, plot_ebc,filter_and_interpolate
from egocentric_head import *
from Egocentric import *
from scipy.stats import weibull_min

def calculate_mr(firing_rates, n, m):
    """
    Calculate the mean resultant vector (MR) for given firing rates.
    
    Args:
        firing_rates (ndarray): Firing rates data.
        n (int): Number of orientation bins.
        m (int): Number of distance bins.
        
    Returns:
        complex: Mean resultant vector (MR).
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    MR = (1 / (n * m)) * np.sum(firing_rates * np.exp(1j * theta[:, None]), axis=(0, 1))
    return MR

def shuffle_spike_train(spike_times, recording_duration, min_shift=30):
    """
    Shuffle spike train by randomly shifting spike times within recording duration.
    
    Args:
        spike_times (array-like): Original spike times.
        recording_duration (float): Total recording duration.
        min_shift (int): Minimum shift interval (default 30).
        
    Returns:
        ndarray: Shuffled spike times.
    """
    # Determine maximum shift based on recording duration
    max_shift = recording_duration - min_shift
    
    # Generate a random shift interval
    shift = np.random.uniform(min_shift, max_shift)
    
    # Shift the spike times
    shuffled_times = (spike_times + shift) % recording_duration
    
    return shuffled_times

def bootstrap_egocentric_body(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, file,speed_threshold, ebc_angle_bin_size,ebc_dist_bin_size, dist_bins):
    """
    Perform bootstrap analysis on egocentric body-centered EBC data and calculate various metrics.
    
    Args:
        dlc_df (DataFrame): Dataframe containing coordinates and timestamps.
        phy_df (DataFrame): Dataframe with spike time data for each cell.
        fps (int): Frames per second of video recording.
        likelihood_threshold (float): Threshold for filtering likelihood data.
        model_dt (float): Time step for interpolating model data.
        bin_width (int): Width of bins for analysis.
        file (str): Filename to save output data.
        speed_threshold (float): Minimum speed threshold for data.
        ebc_angle_bin_size (int): Size of angle bins in degrees.
        ebc_dist_bin_size (int): Size of distance bins.

    Returns:
        tuple: Mean resultant lengths (MRLS), thresholds, mean angles, plot data, distance bins,
               binary plot data, max bins, and preferred distances.
    """
    columns_of_interest = ['neck', 'haunchC', 'time']
    filt_size = 3 # size of bins for gaussian filter
    
    # Adding timestamps to dlc file and only considering columns of interest
    dlc_df['time'] = np.arange(len(dlc_df))/fps

    #filter and interpolate
    model_data_df,model_t = filter_and_interpolate(dlc_df,columns_of_interest,likelihood_threshold,model_dt,fps)

    model_data_df = model_data_df[model_data_df['speed']>speed_threshold]

    neck_x = list(model_data_df['neck x'])
    neck_y = list(model_data_df['neck y'])
    haunchC_x = list(model_data_df['haunchC x'])
    haunchC_y = list(model_data_df['haunchC y'])
    
    egocentric_file = file[:-3]+'ebc_body_data'
    ebc_data = calaculate_ebc(dlc_df, neck_x,neck_y,haunchC_x,haunchC_y, ebc_angle_bin_size, ebc_dist_bin_size)
    
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


    print('bootstrapping ego body...')
    for i in tqdm(cell_numbers):
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

        #"half the arena size" filter
        cell_spikes_avg = cell_spikes_avg[:dist_bins,:]

        cell_spikes_avg = np.divide(cell_spikes_avg,ebc_data_avg)
        
        cell_spikes_avg[np.isnan(cell_spikes_avg)] = 0
        cell_spikes_avg = np.multiply(cell_spikes_avg, fps)

        cell_spikes_avg = cv2.GaussianBlur(cell_spikes_avg,(filt_size,filt_size),filt_size)
        
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

        
        # bootstrap_file = file[:-3]+'_bootstrap_ebc_body_data'
        # if os.path.exists(bootstrap_file+'.npy'):
        #     print('bootstrap file exists')
        #     mrl_thresholds = np.load(bootstrap_file+'.npy')

        # else:
            
        for _ in range(n_bootstrap):

            shuffled_spikes = shuffle_spike_train(spike_times, recording_duration)

            #binning spikes
            sp_count_ind = np.digitize(shuffled_spikes,bins = model_t)

            #-1 because np.digitze is 1-indexed
            sp_count_ind = [i-1 for i in sp_count_ind]

            sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]

            cell_spikes_egocentric = model_data_df['egocentric'].loc[sp_count_ind]  

            cell_spikes_avg = np.sum(cell_spikes_egocentric,axis = 0)

            #"half the arena size" filter
            cell_spikes_avg = cell_spikes_avg[:dist_bins,:]

            cell_spikes_avg = np.divide(cell_spikes_avg,ebc_data_avg)
            
            cell_spikes_avg[np.isnan(cell_spikes_avg)] = 0
            cell_spikes_avg = np.multiply(cell_spikes_avg, fps)
            
            cell_spikes_avg = cv2.GaussianBlur(cell_spikes_avg,(filt_size,filt_size),filt_size)

            shuffled_firing_rates = cell_spikes_avg.copy().T
            theta = abins.copy()
            MR = (1 / (n * m)) * np.sum(shuffled_firing_rates * np.exp(1j * theta[:, None]), axis=(0, 1))
            MRL = np.abs(MR)

            # Append to the shuffled MRLs list
            shuffled_mrls.append(MRL)
        mrl_threshold = np.percentile(shuffled_mrls, 99)
        mrl_thresholds.append(mrl_threshold)
            
    # np.save(bootstrap_file,np.array(mrl_thresholds))


    return MRLS,mrl_thresholds,MALS,ebc_plot_data, distance_bins, ebc_plot_data_binary, max_bins, preferred_dist