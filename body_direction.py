import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from utils.py import set_to_nan_based_on_likelihood, plot_polar_plot


def body_direction(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width):
    
    columns_of_interest = ['center_neck', 'center_haunch', 'time']

    # filtering points based on likelihood and removing likelihood columns
    df_of_interest = dlc_df.filter(regex='|'.join(columns_of_interest))

    # filtering points based on likelihood and removing likelihood columns
    filtered_df = set_to_nan_based_on_likelihood(df_of_interest, likelihood_threshold)
    filtered_df = filtered_df.drop(columns=filtered_df.filter(regex='likelihood').columns)

    ### Make time bins from dlc df
    align_t = filtered_df['time']
    model_t = np.arange(0, np.max(align_t), model_dt)

    model_data = {}
    for col in filtered_df.columns:
        if 'time' not in col:
            interp = interp1d(filtered_df['time'],filtered_df[col],axis=0, bounds_error=False, fill_value='extrapolate')
            model_data[col] = interp(model_t + model_dt / 2)
    

    # Convert model_data to DataFrame for easier handling
    model_data_df = pd.DataFrame(model_data)

    body_angle_rad = np.arctan2((model_data_df['center_neck y'] - model_data_df['center_haunch y']),(model_data_df['center_neck x'] - model_data_df['center_haunch x']))
    #converting to degrees
    body_angle_deg = np.rad2deg(body_angle_rad % (2*np.pi))

    model_data_df['body_angle'] = body_angle_deg

    model_data_df = model_data_df.dropna()

    # Define the bin edges and bin centers
    bin_angles = np.arange(0, 361, bin_width)

    bin_counts, bin_edges = np.histogram(model_data_df['body_angle'], bins=bin_angles)

    #plot_polar_plot(bin_counts,bin_edges)

    cell_numbers = phy_df.index
    plots = []
    for i in cell_numbers:
        spike_times = phy_df.loc[i]['spikeT']

        #removing spike times after camera stopped
        spike_times = spike_times[spike_times<=max(model_t)]

        #binning spikes
        sp_count_ind = np.digitize(spike_times,bins = model_t)

        #-1 because np.digitze is 1-indexed
        sp_count_ind = [i-1 for i in sp_count_ind]

        sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]

        cell_spike_angles = model_data_df['body_angle'][sp_count_ind]

        cell_bin_counts, bin_edges = np.histogram(cell_spike_angles, bins=bin_angles)

        bin_spike_counts_avg = np.round(np.divide(cell_bin_counts,bin_counts),2)

        fig = plot_polar_plot(bin_spike_counts_avg,bin_edges,i)

        plots.append(fig)