import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils import set_to_nan_based_on_likelihood, plot_2d_hist,filter_and_interpolate,corners

def place_cells(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, file, speed_threshold):

    columns_of_interest = ['tailBase', 'time']
    dlc_df['time'] = np.arange(len(dlc_df))/fps

    #filter and interpolate
    model_data_df,model_t = filter_and_interpolate(dlc_df,columns_of_interest,likelihood_threshold,model_dt,fps)

    model_data_df[model_data_df['speed']>speed_threshold]

    br_x = corners('bottom_right x',dlc_df, "bottom_right likelihood", likelihood_threshold)
    bl_x = corners('bottom_left x',dlc_df, "bottom_left likelihood", likelihood_threshold)
    br_y = corners('bottom_right y',dlc_df, "bottom_right likelihood", likelihood_threshold)
    tr_y = corners('top_right y',dlc_df, "top_right likelihood", likelihood_threshold)

    arena_x = 60
    arena_y = 60 
    bin_size_cm = (3,3)

    #height and width of arena
    pixels_x = br_x - bl_x 
    pixels_y = br_y - tr_y

    #pixels to cm
    pixels_per_cm_x = pixels_x/arena_x
    pixels_per_cm_y = pixels_y/arena_y

    #creating bins of arena
    bin_size_pixels = (pixels_per_cm_x*bin_size_cm[0],pixels_per_cm_y*bin_size_cm[1])
    bin_x = int(bin_size_pixels[0])
    x_pos_bins = np.arange(bl_x,br_x,bin_x)
    bin_y = int(bin_size_pixels[1])
    y_pos_bins = np.arange(tr_y,br_y,bin_y)

    #binning mouse position
    bin_counts, x_edges, y_edges = np.histogram2d(model_data_df[columns_of_interest[0]+' x'], model_data_df[columns_of_interest[0]+' y'], bins=[x_pos_bins, y_pos_bins]) 

    cell_numbers = phy_df.index

    plot_data = []
    
    pdf_file = file[:-3]+'_AllocentricPlots.pdf'
    pp = PdfPages(pdf_file)

    for i in cell_numbers:
        spike_times = phy_df.loc[i]['spikeT']

        # Removing spike times after camera stopped
        spike_times = spike_times[spike_times <= max(model_t)]

        # Binning spikes
        sp_count_ind = np.digitize(spike_times, bins=model_t)

        # -1 because np.digitize is 1-indexed
        sp_count_ind = [i - 1 for i in sp_count_ind]

        sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]

        # Get corresponding mouse positions for spike times
        spike_positions_x = model_data_df[columns_of_interest[0] + ' x'][sp_count_ind]
        spike_positions_y = model_data_df[columns_of_interest[0] + ' y'][sp_count_ind]

        # Binning the spike positions in 2D
        spike_bin_counts, _, _ = np.histogram2d(spike_positions_x, spike_positions_y, bins=[x_pos_bins, y_pos_bins])

        # Normalize spike counts by bin counts to get average spike count per bin
        bin_spike_counts_avg = np.round(np.divide(spike_bin_counts, bin_counts, out=np.zeros_like(spike_bin_counts), where=bin_counts != 0), 2)

        fig = plot_2d_hist(bin_spike_counts_avg, x_edges, y_edges,i)

        plot_data.append(bin_spike_counts_avg)
        pp.savefig(fig)
        plt.close(fig)
    
    pp.close()

    return plot_data, x_edges,y_edges

