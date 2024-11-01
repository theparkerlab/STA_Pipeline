import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def set_to_nan_based_on_likelihood(df, threshold):
    for col in df.columns:
        if 'likelihood' in col:
            prefix = col.replace(' likelihood', '')
            df.loc[df[col] < threshold, [f'{prefix} x', f'{prefix} y']] = np.nan
    return df

def plot_polar_plot(bin_counts,bin_edges,i, bin_width):

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Convert bin centers from degrees to radians
    bin_centers_rad = np.deg2rad(bin_centers)

    # Create the bar plot
    bars = ax.bar(bin_centers_rad, bin_counts, width=np.deg2rad(bin_width), edgecolor='k')

    # Add labels and title
    ax.set_title('Cell Number: '+str(i))
    ax.set_theta_zero_location('N')  # Set 0 degrees to the top
    ax.set_theta_direction(1)  # Clockwise

    return fig, ax

def plot_ebc(data,i,distance_bins,angle_bin_size, pixels_per_cm):
    rbins = distance_bins.copy()
    abins = np.linspace(0,2*np.pi, (360//angle_bin_size))

    #calculate histogram
    #hist, _, _ = np.histogram2d(azimut, radius, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)

    # plot
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    fig.suptitle("cell_number: "+(str(i)))

    pc = ax.pcolormesh(A, R, data, cmap="jet")
    ax.set_theta_direction(1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_rticks([0, 200, 400, 600, 800, 1000],labels=np.floor(np.arange(0, 1000 / pixels_per_cm + 1, 200/pixels_per_cm)))  # Less radial ticks
    fig.colorbar(pc)
    return fig

def plot_ebc_head(data,i,distance_bins,angle_bin_size, pixels_per_cm):
    rbins = distance_bins.copy()
    abins = np.linspace(0,2*np.pi, (360//angle_bin_size))

    #calculate histogram
    #hist, _, _ = np.histogram2d(azimut, radius, bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)

    # plot
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    fig.suptitle("cell_number: "+(str(i)))

    pc = ax.pcolormesh(A, R, data, cmap="jet")
    ax.set_theta_direction(1)
    ax.set_theta_offset(np.pi)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=['90°', '135°', '180°', '225°', '270°', '315°', '0°', '45°'])
    ax.set_rticks([0, 200, 400, 600, 800, 1000],labels=np.floor(np.arange(0, 1000 / pixels_per_cm + 1, 200/pixels_per_cm)))  # Less radial ticks
    fig.colorbar(pc)
    return fig

def filter_and_interpolate(dlc_df,columns_of_interest,likelihood_threshold,model_dt,fps):

    # filtering points based on likelihood and removing likelihood columns
    df_of_interest = dlc_df.filter(regex='|'.join(columns_of_interest))

    # filtering points based on likelihood and removing likelihood columns
    filtered_df = set_to_nan_based_on_likelihood(df_of_interest, likelihood_threshold)
    filtered_df = filtered_df.drop(columns=filtered_df.filter(regex='likelihood').columns)

    ### Make time bins from dlc df
    align_t = filtered_df['time']
    model_t = np.arange(0, np.max(align_t), model_dt)

    frame_velocity = speed([columns_of_interest[0]], dlc_df, fps)
    filtered_df['speed'] = frame_velocity


    # Optionally, fill NaNs before interpolation (e.g., forward fill)
    filtered_df = filtered_df.ffill().bfill()

    # Make time bins from dlc df
    align_t = filtered_df['time']
    model_t = np.arange(np.min(align_t), np.max(align_t), model_dt)

    model_data = {}
    for col in filtered_df.columns:
        if 'time' not in col:
            interp = interp1d(filtered_df['time'], filtered_df[col], kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
            model_data[col] = interp(model_t + model_dt / 2)

        # Convert model_data to DataFrame for easier handling
        model_data_df = pd.DataFrame(model_data)

    return model_data_df,model_t

def calcFrameVelocity(keys, dlc_df, fps):
    threshold = 0.95
    for key in keys:
        x_vals = dlc_df[key + ' x'].copy()
        y_vals = dlc_df[key + ' y'].copy()
        x_vals[dlc_df[key + ' likelihood']<threshold] = np.nan
        y_vals[dlc_df[key + ' likelihood']<threshold] = np.nan
        x_vals = interpolateMovement(x_vals)
        y_vals = interpolateMovement(y_vals)

    threshold = 0.9
    top_right_corner_x_vals = dlc_df['top_right_corner x'].copy()
    top_right_corner_x_vals[dlc_df['top_right_corner likelihood']<threshold] = np.nan
    top_right_x = np.nanmean(top_right_corner_x_vals)
    top_left_corner_x_vals = dlc_df['top_left_corner x'].copy()
    top_left_corner_x_vals[dlc_df['top_left_corner likelihood']<threshold] = np.nan
    top_left_x = np.nanmean(top_left_corner_x_vals)
    pixels_per_cm = (top_right_x - top_left_x) / 60
    
    velocityX = np.diff(x_vals/pixels_per_cm) * fps #cm per sec
    velocityY = np.diff(y_vals/pixels_per_cm) * fps #cm per sec
    totalVelocity = np.sqrt((velocityX ** 2) + (velocityY ** 2)) #cm per sec
    totalVelocity[0] = totalVelocity[2]
    totalVelocity[1] = totalVelocity[2]
    totalVelocity[-1] = totalVelocity[-3]
    totalVelocity[-2] = totalVelocity[-3]

    for i in range(0, len(totalVelocity)):
        if totalVelocity[i] > 45:
            totalVelocity[i] = np.nan

    return totalVelocity

def speed(keys, dlc_df, fps):
    threshold = 0.95
    for key in keys:
        x_vals = dlc_df[key + ' x'].copy()
        y_vals = dlc_df[key + ' y'].copy()
        x_vals[dlc_df[key + ' likelihood']<threshold] = np.nan
        y_vals[dlc_df[key + ' likelihood']<threshold] = np.nan
       
        x_vals = interpolateMovement(x_vals)
       
        y_vals = interpolateMovement(y_vals)

    threshold = 0.9
    top_right_corner_x_vals = dlc_df['top_right_corner x'].copy()
    top_right_corner_x_vals[dlc_df['top_right_corner likelihood']<threshold] = np.nan
    top_right_x = np.nanmean(top_right_corner_x_vals)
    top_left_corner_x_vals = dlc_df['top_left_corner x'].copy()
    top_left_corner_x_vals[dlc_df['top_left_corner likelihood']<threshold] = np.nan
    top_left_x = np.nanmean(top_left_corner_x_vals)
    pixels_per_cm = (top_right_x - top_left_x) / 60
    
    velocityX = np.diff(x_vals/pixels_per_cm) * fps #cm per sec
    velocityY = np.diff(y_vals/pixels_per_cm) * fps #cm per sec
    totalVelocity = np.sqrt((velocityX ** 2) + (velocityY ** 2)) #cm per sec
    totalVelocity[0] = totalVelocity[2]
    totalVelocity[1] = totalVelocity[2]
    totalVelocity[-1] = totalVelocity[-3]
    totalVelocity[-2] = totalVelocity[-3]

    for i in range(0, len(totalVelocity)):
        if totalVelocity[i] > 45:
            totalVelocity[i] = np.nan
    
    totalVelocity=  np.append(totalVelocity,totalVelocity[-1])

    return totalVelocity


def interpolateMovement(arr):
    arr = pd.Series(arr).interpolate().to_numpy()
    arr = pd.Series(arr).fillna(method='bfill').to_numpy()
    arr = signal.medfilt(arr,kernel_size = 5)
    box_size = 5
    box = np.ones(box_size) / box_size
    arr = np.convolve(arr, box, mode='same')
    return arr

def removeWhenStationary(spikeTimes, model_t, frame_velocity):
    intervalSize = model_t[1]
    speedThreshold = 1
    maxTime = model_t[1]
    for i in range (0, len(spikeTimes)):
        currentSpike = spikeTimes[i]
        while currentSpike > maxTime:
            minTime+=intervalSize
        timeIndex = model_t.index(minTime)
        if frame_velocity[timeIndex] < speedThreshold:
            spikeTimes[i] = np.nan
    spikeTimes = spikeTimes[~np.isnan(spikeTimes)] 
    return spikeTimes

def corners(point,dlc_df, point_likelihood, likelihood_threshold):
    vals = dlc_df[point].copy()
    vals[dlc_df[point_likelihood]<likelihood_threshold] = np.nan
    return np.nanmean(vals)

def plot_2d_hist(bin_counts, x_edges,y_edges):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(bin_counts, origin='lower', aspect='auto',
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.colorbar(label='spikes')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Histogram of spikes')
    return fig