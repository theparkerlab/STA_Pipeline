'''
computes egocentric head-centered data by calculating egocentric boundary cell (EBC) bins and spike data related to an animal’s head orientation. It filters, interpolates, and organizes the data by distance and angle bins, generating visualizations and saving the processed data as .npy and .pdf files for further analysis.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import os
import ray
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils import set_to_nan_based_on_likelihood, plot_ebc_head,filter_and_interpolate

def ebc_bins(dlc_df, bin_size_angle=12,bin_size_distance=160):
    """
    Calculate angle and distance bins for egocentric boundary cell (EBC) analysis.
    
    Args:
        dlc_df (DataFrame): Dataframe containing corner coordinates of the arena.
        bin_size_angle (int): Size of angle bins in degrees.
        bin_size_distance (int): Size of distance bins.

    Returns:
        tuple: Distance and angle bins as numpy arrays.
    """ 
    angle_bins = np.arange(0,360,bin_size_angle)

    top_right_corner_x = dlc_df[dlc_df['top_right likelihood'] > 0.95]['top_right x'].median()
    top_right_corner_y = dlc_df[dlc_df['top_right likelihood'] > 0.95]['top_right y'].median()

    bottom_left_corner_x = dlc_df[dlc_df['bottom_left likelihood'] > 0.95]['bottom_left x'].median()
    bottom_left_corner_y = dlc_df[dlc_df['bottom_left likelihood'] > 0.95]['bottom_left y'].median()

    diagonal_distance_arena = math.hypot(top_right_corner_x - bottom_left_corner_x, top_right_corner_y - bottom_left_corner_y)
    distance_bins = np.arange(0,diagonal_distance_arena,bin_size_distance)
    return distance_bins,angle_bins

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Args:
        origin (tuple): Origin point for rotation (x, y).
        point (tuple): Point to rotate (x, y).
        angle (float): Angle in radians for rotation.

    Returns:
        tuple: Coordinates of the rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def lineLineIntersection(A, B, C, D):
    """
    Find the intersection point of two lines, AB and CD.

    Args:
        A, B (tuple): Endpoints of the first line.
        C, D (tuple): Endpoints of the second line.

    Returns:
        tuple or str: Intersection point (x, y) or 'parallel' if lines are parallel.
    """
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*(A[0]) + b1*(A[1])
 
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*(C[0]) + b2*(C[1])
 
    determinant = a1*b2 - a2*b1
 
    if (determinant == 0):
        # The lines are parallel. This is simplified
        # by returning a pair of FLT_MAX
        return 'parallel'
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return (x, y)


def calaculate_ebc_head(dlc_df,driveL_x,driveL_y,driveR_x,driveR_y,ebc_angle_bin_size,ebc_dist_bin_size):
    """
    Calculate EBC data for each frame based on head orientation and arena boundaries.

    Args:
        dlc_df (DataFrame): Dataframe containing corner coordinates of the arena.
        driveL_x, driveL_y (list): X, Y coordinates for the left side of head.
        driveR_x, right_dirve_y (list): X, Y coordinates for the right side of head.
        ebc_angle_bin_size (int): Size of angle bins in degrees.
        ebc_dist_bin_size (int): Size of distance bins.

    Returns:
        ndarray: Array of EBC data for each frame.
    """
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    top_left_corner = (dlc_df[dlc_df['top_left likelihood'] > 0.95]['top_left x'].median(), dlc_df[dlc_df['top_left likelihood'] > 0.95]['top_left y'].median())
    top_right_corner = (dlc_df[dlc_df['top_right likelihood'] > 0.95]['top_right x'].median(), dlc_df[dlc_df['top_right likelihood'] > 0.95]['top_right y'].median())
    bottom_left_corner = (dlc_df[dlc_df['bottom_left likelihood'] > 0.95]['bottom_left x'].median(), dlc_df[dlc_df['bottom_left likelihood'] > 0.95]['bottom_left y'].median())
    bottom_right_corner = (dlc_df[dlc_df['bottom_right likelihood'] > 0.95]['bottom_right x'].median(), dlc_df[dlc_df['bottom_right likelihood'] > 0.95]['bottom_right y'].median())
    distance_bins,angle_bins = ebc_bins(dlc_df, ebc_angle_bin_size,ebc_dist_bin_size)
    ebc_data_final = []
    futures = []

    # print(len(driveL_x))
    for i in range(0, len(driveL_x), 1000):
        futures.append(process_egocentric_head.remote(i, min(i + 1000, len(driveL_x)), driveL_x, driveL_y, driveR_x, driveR_y, ebc_angle_bin_size, ebc_dist_bin_size, distance_bins, angle_bins, top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner))
    ebc_data_final = ray.get(futures)

    flattened_array = np.concatenate([batch for batch in ebc_data_final])
    # print(flattened_array.shape)
    ray.shutdown()
    return flattened_array

@ray.remote
def process_egocentric_head(startIndex, stopIndex, driveL_x, driveL_y, driveR_x, driveR_y, ebc_angle_bin_size, ebc_dist_bin_size, distance_bins, angle_bins, top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner):
    ebc_data_batch = []
    for i in range(startIndex, stopIndex):
        ebc_bins_total = np.zeros((len(distance_bins),len(angle_bins)))
        for angle in range(0,360,ebc_angle_bin_size):
            center_neck_pos = (driveL_x[i],driveL_y[i])
            center_haunch_pos = (driveR_x[i],driveR_y[i]) 
            center_neck_pos = rotate(center_haunch_pos,center_neck_pos,angle=math.radians(-1*angle))
            body_angle_radian_frame = math.atan2(center_haunch_pos[1]-center_neck_pos[1],center_haunch_pos[0]-center_neck_pos[0])

            body_angle_deg_frame = math.degrees(body_angle_radian_frame)

            if body_angle_deg_frame<0:
                body_angle_deg_frame = 360+body_angle_deg_frame
            

            if(body_angle_deg_frame==0):
                #left wall
                interpoint = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_left_corner,top_left_corner)
                min_distance = math.hypot(interpoint[0]-center_neck_pos[0],interpoint[1]-center_neck_pos[1])
                distance_bin_index = np.digitize(min_distance,distance_bins)
                angle_bin_index = np.digitize(angle,angle_bins)
                ebc_bins_total[distance_bin_index-1][angle_bin_index]+=1

            elif(body_angle_deg_frame>0 and body_angle_deg_frame<90):
                #left wall and top wall
                interpoint_l = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_left_corner,top_left_corner)
                interpoint_t = lineLineIntersection(center_haunch_pos,center_neck_pos,top_left_corner,top_right_corner)
                distance_from_point_l = math.hypot(interpoint_l[0]-center_neck_pos[0],interpoint_l[1]-center_neck_pos[1])
                distance_from_point_t = math.hypot(interpoint_t[0]-center_neck_pos[0],interpoint_t[1]-center_neck_pos[1])
                min_distance = min(distance_from_point_l,distance_from_point_t)
                distance_bin_index = np.digitize(min_distance,distance_bins)
                angle_bin_index = np.digitize(angle,angle_bins)
                ebc_bins_total[distance_bin_index-1][angle_bin_index-1]+=1
            
            elif(body_angle_deg_frame==90):
                #top wall
                interpoint = lineLineIntersection(center_haunch_pos,center_neck_pos,top_left_corner,top_right_corner)
                min_distance = math.hypot(interpoint[0]-center_neck_pos[0],interpoint[1]-center_neck_pos[1])
                distance_bin_index = np.digitize(min_distance,distance_bins)
                angle_bin_index = np.digitize(angle,angle_bins)
                ebc_bins_total[distance_bin_index-1][angle_bin_index-1]+=1

            elif(body_angle_deg_frame>90 and body_angle_deg_frame<180):
                #top wall and right wall
                interpoint_l = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_right_corner,top_right_corner)
                interpoint_t = lineLineIntersection(center_haunch_pos,center_neck_pos,top_left_corner,top_right_corner)
                distance_from_point_l = math.hypot(interpoint_l[0]-center_neck_pos[0],interpoint_l[1]-center_neck_pos[1])
                distance_from_point_t = math.hypot(interpoint_t[0]-center_neck_pos[0],interpoint_t[1]-center_neck_pos[1])
                min_distance = min(distance_from_point_l,distance_from_point_t)
                distance_bin_index = np.digitize(min_distance,distance_bins)
                angle_bin_index = np.digitize(angle,angle_bins)
                ebc_bins_total[distance_bin_index-1][angle_bin_index-1]+=1
                
            elif(body_angle_deg_frame==180):
                #right wall
                interpoint = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_right_corner,top_right_corner)
                min_distance = math.hypot(interpoint[0]-center_neck_pos[0],interpoint[1]-center_neck_pos[1])
                distance_bin_index = np.digitize(min_distance,distance_bins)
                angle_bin_index = np.digitize(angle,angle_bins) 
                ebc_bins_total[distance_bin_index-1][angle_bin_index-1]+=1

            elif(body_angle_deg_frame>180 and body_angle_deg_frame<270):
                #right wall and bottom wall
                
                interpoint_l = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_right_corner,top_right_corner)
                interpoint_t = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_left_corner,bottom_right_corner)
                distance_from_point_l = math.hypot(interpoint_l[0]-center_neck_pos[0],interpoint_l[1]-center_neck_pos[1])
                distance_from_point_t = math.hypot(interpoint_t[0]-center_neck_pos[0],interpoint_t[1]-center_neck_pos[1])
                min_distance = min(distance_from_point_l,distance_from_point_t)
                distance_bin_index = np.digitize(min_distance,distance_bins)
                angle_bin_index = np.digitize(angle,angle_bins)
                ebc_bins_total[distance_bin_index-1][angle_bin_index-1]+=1
            
            elif(body_angle_deg_frame == 270):
                #bottom wall
                interpoint = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_right_corner,bottom_left_corner)
                min_distance = math.hypot(interpoint[0]-center_neck_pos[0],interpoint[1]-center_neck_pos[1])
                distance_bin_index = np.digitize(min_distance,distance_bins)
                angle_bin_index = np.digitize(angle,angle_bins)
                ebc_bins_total[distance_bin_index-1][angle_bin_index-1]+=1
            
            else:
                #bottom wall and left wall
                interpoint_l = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_left_corner,top_left_corner)
                interpoint_t = lineLineIntersection(center_haunch_pos,center_neck_pos,bottom_left_corner,bottom_right_corner)
                distance_from_point_l = math.hypot(interpoint_l[0]-center_neck_pos[0],interpoint_l[1]-center_neck_pos[1])
                distance_from_point_t = math.hypot(interpoint_t[0]-center_neck_pos[0],interpoint_t[1]-center_neck_pos[1])
                min_distance = min(distance_from_point_l,distance_from_point_t)
                distance_bin_index = np.digitize(min_distance,distance_bins)
                if min_distance<=800:
                    angle_bin_index = np.digitize(angle,angle_bins)
                    #print(i,min_distance,distance_bin_index,angle_bin_index)
                    ebc_bins_total[distance_bin_index-1][angle_bin_index-1]+=1
        ebc_data_batch.append(ebc_bins_total)
    return ebc_data_batch

def egocentric_head(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, file, speed_threshold, ebc_angle_bin_size,ebc_dist_bin_size, dist_bins):
    """
    Compute egocentric head-centered (EBC) data and generate plots for head-related spikes.

    Args:
        dlc_df (DataFrame): Dataframe containing coordinates and timestamps for arena.
        phy_df (DataFrame): Spike time data for each cell.
        fps (int): Frames per second of the video.
        likelihood_threshold (float): Minimum likelihood for data to be considered valid.
        model_dt (float): Time step for interpolating model data.
        bin_width (int): Width of the bins.
        file (str): Filename for saving results.
        speed_threshold (float): Minimum speed threshold for filtering.
        ebc_angle_bin_size (int): Size of angle bins in degrees.
        ebc_dist_bin_size (int): Size of distance bins.

    Returns:
        tuple: EBC data for plotting, distance bins, binary EBC data, and max bin locations.
    """

    columns_of_interest = ['driveL','driveR', 'time']

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
    distance_bins = distance_bins[:dist_bins] #cut off far half of the arena
    rbins = distance_bins.copy()
    abins = np.linspace(0,2*np.pi, (360//ebc_angle_bin_size))

    model_data_df['egocentric'] = list(ebc_data)

    cell_numbers = phy_df.index

    spike_data_head = []
    max_bins = []

    ebc_plot_data = []
    ebc_plot_data_binary = []

    for i in cell_numbers:
        spike_times = phy_df.loc[i]['spikeT']

        #removing spike times after camera stopped
        spike_times = spike_times[spike_times<=max(model_t)]

        #binning spikes
        sp_count_ind = np.digitize(spike_times,bins = model_t)

        #-1 because np.digitize is 1-indexed
        sp_count_ind = [i-1 for i in sp_count_ind]

        sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]

        cell_spikes_egocentric = model_data_df['egocentric'].loc[sp_count_ind]
        cell_spikes_egocentric = cell_spikes_egocentric.apply(lambda x: x[:dist_bins, :]) #truncates by :dist_bins

        print(cell_spikes_egocentric.iloc[0].shape, "\n\n")

        cell_spikes_avg = np.sum(cell_spikes_egocentric,axis = 0)
        cell_spikes_avg = np.divide(cell_spikes_avg,ebc_data_avg)

        cell_spikes_avg[np.isnan(cell_spikes_avg)] = 0
        cell_spikes_avg = np.multiply(cell_spikes_avg, fps)

        ebc_plot_data.append(cell_spikes_avg)

        arr = np.zeros(len(model_t))

        # Use np.add.at for efficient indexing
        np.add.at(arr, sp_count_ind, 1)

        spike_data_head.append(arr)

        #75% threshold    
        max_idx = np.unravel_index(np.argmax(cell_spikes_avg, axis=None), cell_spikes_avg.shape)

        # Corresponding radius and angle for the maximum value
        max_radius = rbins[max_idx[0]]
        max_angle = abins[max_idx[1]]

        threshold = np.percentile(cell_spikes_avg, 75)

        # Convert the array into a binary array
        binary_array = np.where(cell_spikes_avg >= threshold, 1, 0)

        ebc_plot_data_binary.append(binary_array)

        max_bins.append([max_angle,max_radius])

    
    plots = []
    pdf_file = file[:-3]+'_ebc_headPlots.pdf'
    pp = PdfPages(pdf_file)
    pixels_per_cm = (dlc_df[dlc_df['top_right likelihood'] > 0.95]['top_right x'].median() - dlc_df[dlc_df['top_left likelihood'] > 0.95]['top_left x'].median()) / 60

    for i in range(len(ebc_plot_data)):
        fig = plot_ebc_head(ebc_plot_data[i][:dist_bins],i, distance_bins[:dist_bins],ebc_angle_bin_size, pixels_per_cm)
        plots.append(fig)
        pp.savefig(fig)

    pp.close()
    
    spike_data_head = np.array(spike_data_head)

    np.save('spike_data_head',spike_data_head)

    np.save('egocentric_data_head_4x',np.array(ebc_data))

    return ebc_plot_data, distance_bins,ebc_plot_data_binary, max_bins
    

