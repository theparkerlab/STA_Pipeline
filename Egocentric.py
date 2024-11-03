'''
Analyzes egocentric body-centered data by calculating egocentric boundary cell (EBC) bins and spike data in relation to an animal's body and neck orientation. It computes distance and angle bins, filters the data, and generates visualizations, saving the processed data and plots as .npy and .pdf files for further analysis.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import matplotlib
import os
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils import set_to_nan_based_on_likelihood, plot_ebc,filter_and_interpolate

def ebc_bins(dlc_df, bin_size_angle=12,bin_size_distance=160):
    """
    Calculate angle and distance bins for egocentric boundary cell (EBC) analysis.

    Args:
        dlc_df (DataFrame): Dataframe containing coordinate information of the arena.
        bin_size_angle (int): Size of angle bins in degrees.
        bin_size_distance (int): Size of distance bins.

    Returns:
        tuple: Distance and angle bins as numpy arrays.
    """ 
    top_left_corner = (dlc_df.iloc[0]['top_left_corner x'], dlc_df.iloc[0]['top_left_corner y'])
    top_right_corner = (dlc_df.iloc[0]['top_right_corner x'],dlc_df.iloc[0]['top_right_corner y'])
    bottom_left_corner = (dlc_df.iloc[0]['bottom_left_corner x'], dlc_df.iloc[0]['bottom_left_corner y'])
    bottom_right_corner = (dlc_df.iloc[0]['bottom_right_corner x'], dlc_df.iloc[0]['bottom_right_corner y'])
    angle_bins = np.arange(0,360,bin_size_angle)

    diagonal_distance_arena = math.hypot(top_right_corner[0] - bottom_left_corner[0], top_right_corner[1] - bottom_left_corner[1])
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

def calaculate_ebc(dlc_df,center_neck_x,center_neck_y,center_haunch_x,center_haunch_y, ebc_angle_bin_size, ebc_dist_bin_size):
    """
    Calculate EBC data for each frame based on body orientation and arena boundaries.

    Args:
        dlc_df (DataFrame): Dataframe containing coordinate data for the arena.
        center_neck_x, center_neck_y (list): Lists of x, y coordinates for the neck.
        center_haunch_x, center_haunch_y (list): Lists of x, y coordinates for the haunch.
        ebc_angle_bin_size (int): Size of angle bins in degrees.
        ebc_dist_bin_size (int): Size of distance bins.

    Returns:
        ndarray: Array of EBC data for each frame.
    """
    top_left_corner = (dlc_df.iloc[0]['top_left_corner x'], dlc_df.iloc[0]['top_left_corner y'])
    top_right_corner = (dlc_df.iloc[0]['top_right_corner x'],dlc_df.iloc[0]['top_right_corner y'])
    bottom_left_corner = (dlc_df.iloc[0]['bottom_left_corner x'], dlc_df.iloc[0]['bottom_left_corner y'])
    bottom_right_corner = (dlc_df.iloc[0]['bottom_right_corner x'], dlc_df.iloc[0]['bottom_right_corner y'])
    distance_bins,angle_bins = ebc_bins(dlc_df, ebc_angle_bin_size, ebc_dist_bin_size)
    ebc_data_final = []
    for i in range(len(center_neck_x)):
        print(i, len(center_neck_x), "egocentric body") #TO VIEW PROGRESS
        ebc_bins_total = np.zeros((len(distance_bins),len(angle_bins)))
        for angle in range(0,360,3):
            #center_neck_pos = (frame['center_neck_x'],frame['center_neck_y'])
            #center_haunch_pos = (frame['center_haunch_x'],frame['center_haunch_y'])
            center_neck_pos = (center_neck_x[i],center_neck_y[i])
            center_haunch_pos = (center_haunch_x[i],center_haunch_y[i]) 
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
        
        ebc_data_final.append(ebc_bins_total)
    return np.array(ebc_data_final)

def egocentric_body(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, file,speed_threshold, ebc_angle_bin_size,ebc_dist_bin_size):
    """
    Compute egocentric body-centered (EBC) data and generate plots for body-related spikes.

    Args:
        dlc_df (DataFrame): Coordinate and time data for the arena.
        phy_df (DataFrame): Spike time data for each cell.
        fps (int): Frames per second of the video.
        likelihood_threshold (float): Minimum likelihood for data to be considered valid.
        model_dt (float): Time step for interpolating model data.
        bin_width (int): Width of the bins.
        file (str): Filename for saving results.
        speed_threshold (float): Minimum speed threshold.
        ebc_angle_bin_size (int): Size of angle bins in degrees.
        ebc_dist_bin_size (int): Size of distance bins.

    Returns:
        tuple: EBC data for plotting, distance bins, binary EBC data, and max bin locations.
    """

    columns_of_interest = ['center_neck', 'center_haunch', 'time']
    
    # Adding timestamps to dlc file and only considering columns of interest
    dlc_df['time'] = np.arange(len(dlc_df))/fps

    #filter and interpolate
    model_data_df,model_t = filter_and_interpolate(dlc_df,columns_of_interest,likelihood_threshold,model_dt,fps)

    model_data_df = model_data_df[model_data_df['speed']>speed_threshold]

    center_neck_x = list(model_data_df['center_neck x'])
    center_neck_y = list(model_data_df['center_neck y'])
    center_haunch_x = list(model_data_df['center_haunch x'])
    center_haunch_y = list(model_data_df['center_haunch y'])
    
    egocentric_file = file[:-3]+'ebc_body_data'
    if os.path.exists(egocentric_file+'.npy'):
        ebc_data = np.load(egocentric_file+'.npy')
    else:
        ebc_data = calaculate_ebc(dlc_df, center_neck_x,center_neck_y,center_haunch_x,center_haunch_y, ebc_angle_bin_size, ebc_dist_bin_size)
        np.save(egocentric_file,np.array(ebc_data))

    distance_bins,angle_bins = ebc_bins(dlc_df, ebc_angle_bin_size, ebc_dist_bin_size)

    ebc_data_avg = np.sum(ebc_data,axis=0)
    rbins = distance_bins.copy()
    abins = np.linspace(0,2*np.pi, 121)

    model_data_df['egocentric'] = list(ebc_data)

    cell_numbers = phy_df.index
    

    ebc_plot_data = []
    ebc_plot_data_binary = []

    spike_data_body = []
    max_bins = []

    for i in cell_numbers:
        spike_times = phy_df.loc[i]['spikeT']

        #removing spike times after camera stopped
        spike_times = spike_times[spike_times<=max(model_t)]

        #binning spikes
        sp_count_ind = np.digitize(spike_times,bins = model_t)

        #-1 because np.digitze is 1-indexed
        sp_count_ind = [i-1 for i in sp_count_ind]

        sp_count_ind = [i for i in sp_count_ind if i in model_data_df.index]

        #grouping egocentric data based on spikes
        cell_spikes_egocentric = model_data_df['egocentric'].loc[sp_count_ind]  

        cell_spikes_avg = np.sum(cell_spikes_egocentric,axis = 0)
        cell_spikes_avg = np.divide(cell_spikes_avg,ebc_data_avg)
        
        cell_spikes_avg[np.isnan(cell_spikes_avg)] = 0
        cell_spikes_avg = np.multiply(cell_spikes_avg, fps)

        ebc_plot_data.append(cell_spikes_avg)
        
        arr = np.zeros(len(model_t))

        # Use np.add.at for efficient indexing
        np.add.at(arr, sp_count_ind, 1)

        spike_data_body.append(arr)

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
    pdf_file = file[:-3]+'_ebc_bodyPlots.pdf'
    pp = PdfPages(pdf_file)
    pixels_per_cm = (dlc_df.iloc[0]['top_right_corner x'] - dlc_df.iloc[0]['top_left_corner x']) / 60


    for i in range(len(ebc_plot_data)):
        fig = plot_ebc(ebc_plot_data[i],i, distance_bins, ebc_angle_bin_size, pixels_per_cm)
        plots.append(fig)
        pp.savefig(fig)

    pp.close()

    spike_data_body = np.array(spike_data_body)

    np.save('spike_data_body',spike_data_body)

    np.save('egocentric_data_body',np.array(ebc_data))
    
    return ebc_plot_data, distance_bins, ebc_plot_data_binary, max_bins
    
