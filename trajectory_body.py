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

def trajectory_body(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, file, speed_threshold):

    columns_of_interest = ['center_haunch','center_neck','time']

    # Adding timestamps to dlc file and only considering columns of interest
    dlc_df['time'] = np.arange(len(dlc_df))/fps

    #filter and interpolate
    model_data_df,model_t = filter_and_interpolate(dlc_df,columns_of_interest,likelihood_threshold,model_dt,fps)

    model_data_df = model_data_df[model_data_df['speed']>speed_threshold]

    bds = []
    y = list(model_data_df['center_haunch y']-model_data_df['center_neck y'])
    x = list(model_data_df['center_haunch x']-model_data_df['center_neck x'])
    for i in range(len(x)):
        bds.append(math.atan2(y[i],x[i]))

    model_data_df['head_direction'] = bds

    trajectory_df = dlc_df[['center_haunch x', 'center_haunch y']]

    trajectory_df = trajectory_df[trajectory_df['center_haunch x']>400]
    trajectory_df = trajectory_df[trajectory_df['center_haunch x']<1200]
    trajectory_df = trajectory_df[trajectory_df['center_haunch y']>100]
    trajectory_df = trajectory_df[trajectory_df['center_haunch y']<900]

    ch_points_x = list(trajectory_df['center_haunch x'])
    ch_points_y = list(trajectory_df['center_haunch y'])

    cell_numbers = phy_df.index
    
    mouse_xsb = []
    mouse_ysb = []
    cell_bds = []
    for i in cell_numbers:
        spike_times = phy_df.loc[i]['spikeT']
        spike_times = spike_times[spike_times <= max(model_t)]

        # Binning spikes
        sp_count_ind = np.digitize(spike_times, bins=model_t)

        # -1 because np.digitize is 1-indexed
        sp_count_ind = [i - 1 for i in sp_count_ind]

        sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]



        # Get corresponding mouse positions for spike times
        mouse_x = list(model_data_df[columns_of_interest[0] + ' x'][sp_count_ind])
        mouse_y = list(model_data_df[columns_of_interest[0] + ' y'][sp_count_ind])
        head_direction = np.rad2deg(list(model_data_df['head_direction'][sp_count_ind]))

        mask = (np.array(mouse_x) >= 400) & (np.array(mouse_x) <= 1200) & (np.array(mouse_y) >= 100) & (np.array(mouse_y) <= 900)

        # Apply the mask to limit the data
        mouse_x = np.array(mouse_x)[mask]
        mouse_y = np.array(mouse_y)[mask]
        head_direction = np.array(head_direction)[mask]

        mouse_xsb.append(mouse_x)
        mouse_ysb.append(mouse_y)
        cell_bds.append(head_direction)
    
    
    return mouse_xsb,mouse_ysb,cell_bds,ch_points_x,ch_points_y