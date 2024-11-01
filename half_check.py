import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils import set_to_nan_based_on_likelihood, plot_ebc, filter_and_interpolate
from egocentric_head import *
from Egocentric import *
from scipy.stats import weibull_min



def process_half(dlc_df_half, columns_of_interest, likelihood_threshold, model_dt, fps, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size):
    # Filter and interpolate
    model_data_df, model_t = filter_and_interpolate(dlc_df_half, columns_of_interest, likelihood_threshold, model_dt, fps)
    
    # Filter based on speed threshold
    model_data_df = model_data_df[model_data_df['speed'] > speed_threshold]
    
    # Get coordinates for ebc calculation
    center_neck_x = list(model_data_df['left_drive x'])
    center_neck_y = list(model_data_df['left_drive y'])
    center_haunch_x = list(model_data_df['right_drive x'])
    center_haunch_y = list(model_data_df['right_drive y'])
    
    # Calculate ebc
    ebc_data = calaculate_ebc_head(dlc_df_half, center_neck_x, center_neck_y, center_haunch_x, center_haunch_y, ebc_angle_bin_size, ebc_dist_bin_size)
    distance_bins, angle_bins = ebc_bins(dlc_df_half, ebc_angle_bin_size, ebc_dist_bin_size)
    
    ebc_data_avg = np.sum(ebc_data, axis=0)
    rbins = distance_bins.copy()
    abins = np.linspace(0, 2*np.pi, 121)
    
    model_data_df['egocentric'] = list(ebc_data)
    
    return model_data_df, model_t

def calc_mrls(model_data_df, phy_df, cell_numbers, model_t, abins):
    ebc_plot_data = []
    n = 120
    m = 27
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

        ebc_data_avg = np.sum(np.array(model_data_df['egocentric']), axis=0)
        cell_spikes_avg = np.divide(cell_spikes_avg, ebc_data_avg)
        
        cell_spikes_avg[np.isnan(cell_spikes_avg)] = 0
        
        ebc_plot_data.append(cell_spikes_avg)

        firing_rates = cell_spikes_avg.copy().T
        theta = abins.copy()
        MR = (1 / (n * m)) * np.sum(firing_rates * np.exp(1j * theta[:, None]), axis=(0, 1))
        MRL = np.abs(MR)
        MRA = np.angle(MR)

        MRLS.append(MRL)
        MALS.append(MRA)
        preferred_orientation_idx = np.argmin(np.abs(theta - MRA))
        firing_rate_vector = firing_rates[preferred_orientation_idx, :]

        # Fit a Weibull distribution
        params = weibull_min.fit(firing_rate_vector)

        # Get the distance bin with the maximum estimated firing rate
        max_firing_distance_bin = np.argmax(weibull_min.pdf(np.arange(m), *params))

        preferred_dist.append(max_firing_distance_bin)
    return MRLS, MALS,preferred_dist


def egocentric_head_half_check(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, file, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size):

    columns_of_interest = ['left_drive', 'right_drive', 'time']

    # Adding timestamps to dlc file and only considering columns of interest
    dlc_df['time'] = np.arange(len(dlc_df)) / fps

    # Split dlc_df into two halves
    half_len = len(dlc_df) // 2
    dlc_df_1 = dlc_df.iloc[:half_len].copy()
    dlc_df_2 = dlc_df.iloc[half_len:].copy()

    # Process each half
    model_data_df_1, model_t1 = process_half(dlc_df_1, columns_of_interest, likelihood_threshold, model_dt, fps, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size)
    model_data_df_2, model_t2 = process_half(dlc_df_2, columns_of_interest, likelihood_threshold, model_dt, fps, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size)

    cell_numbers = phy_df.index
    distance_bins, angle_bins = ebc_bins(dlc_df, ebc_angle_bin_size, ebc_dist_bin_size)
    rbins = distance_bins.copy()
    abins = np.linspace(0, 2*np.pi, (360 // ebc_angle_bin_size))

    half_check_file = file[:-3]+'_half_ebc_head_data'
    if os.path.exists(half_check_file+'.npy'):
        print('half_check file exists')
        MRLS_1, MRLS_2, MALS_1, MALS_2 = np.load(half_check_file+'.npy')
    else:
        MRLS_1, MALS_1,pref_dist_1 = calc_mrls(model_data_df_1, phy_df, cell_numbers, model_t1, abins)
        MRLS_2, MALS_2,pref_dist_2 = calc_mrls(model_data_df_2, phy_df, cell_numbers, model_t2, abins)
        np.save(half_check_file,np.array([MRLS_1, MRLS_2, MALS_1, MALS_2,pref_dist_1,pref_dist_2]))

    return MRLS_1, MRLS_2, MALS_1, MALS_2,pref_dist_1,pref_dist_2
