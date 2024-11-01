import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils import set_to_nan_based_on_likelihood, plot_polar_plot,filter_and_interpolate

def head_direction(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, file, speed_threshold):

    columns_of_interest = ['left_drive', 'right_drive', 'time']
    dlc_df['time'] = np.arange(len(dlc_df))/fps

    #filter and interpolate
    model_data_df,model_t = filter_and_interpolate(dlc_df,columns_of_interest,likelihood_threshold,model_dt,fps)

    model_data_df[model_data_df['speed']>speed_threshold]

    head_angle_rad = np.add(np.arctan2((model_data_df['left_drive y'] - model_data_df['right_drive y']),(model_data_df['left_drive x'] - model_data_df['right_drive x'])),np.deg2rad(90))
    #converting to degrees
    head_angle_deg = np.rad2deg(head_angle_rad % (2*np.pi))

    model_data_df['head_angle'] = head_angle_deg

    model_data_df = model_data_df.dropna()
    # Define the bin edges and bin centers

    bin_angles = np.arange(0, 361, bin_width)

    bin_counts, bin_edges = np.histogram(model_data_df['head_angle'], bins=bin_angles)

    cell_numbers = phy_df.index
    
    plot_data = []
    
    pdf_file = file[:-3]+'_headPlots.pdf'
    pp = PdfPages(pdf_file)

    for i in cell_numbers:
        spike_times = phy_df.loc[i]['spikeT'] 

        #removing spike times after camera stopped
        spike_times = spike_times[spike_times<=max(model_t)]

        #binning spikes
        sp_count_ind = np.digitize(spike_times,bins = model_t)

        #-1 because np.digitze is 1-indexed
        sp_count_ind = [i-1 for i in sp_count_ind]

        #to account for dropped na values since they wont be in the df
        sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]
        
        cell_spike_angles = model_data_df['head_angle'][sp_count_ind]

        cell_bin_counts, bin_edges = np.histogram(cell_spike_angles, bins=bin_angles)

        bin_spike_counts_avg = np.round(np.divide(cell_bin_counts,bin_counts),2)
        bin_spike_counts_avg = np.multiply(bin_spike_counts_avg, fps)

        fig, ax = plot_polar_plot(bin_spike_counts_avg,bin_edges,i, bin_width)

        plot_data.append(bin_spike_counts_avg)
        pp.savefig(fig)
    
    pp.close()

    return plot_data, bin_edges