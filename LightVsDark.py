#Imports
import os, fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import cv2
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askdirectory

from speed import speedPlots
from body_direction import body_direction
from Egocentric import *
from Head_direction import *
from egocentric_head import *
from place_cells import *
from utils import *

# Find file
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files: 
            if fnmatch.fnmatch(name,pattern): 
                result.append(os.path.join(root,name))
    if len(result)==1:
        result = result[0]
    return result

#find directory
def findDir(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in dirs: 
            if fnmatch.fnmatch(name,pattern): 
                result.append(os.path.join(root,name))
    if len(result)==1:
        result = result[0]
    return result


# File/Video Initialization
root = Tk()
root.withdraw() 
path = askdirectory(title='Choose experiment folder', initialdir=r'\\rhea\E\ephys') # show an "Open" dialog box and return the path to the selected file
print('you have selected: ', path)

lightPath = findDir("FM_LIGHT", path)
print(lightPath)
dlc_phy_file_light = find('*topDLCephys.h5',lightPath)

dlc_df_light = pd.read_hdf(dlc_phy_file_light, 'dlc_df')
phy_df_light = pd.read_hdf(dlc_phy_file_light,'phy_df')


darkPath = findDir("FM_DARK", path)
print(darkPath)
dlc_phy_file_light = find('*topDLCephys.h5',lightPath)

dlc_phy_file_dark = find('*topDLCephys.h5',darkPath) 

dlc_df_dark = pd.read_hdf(dlc_phy_file_dark, 'dlc_df')
phy_df_dark = pd.read_hdf(dlc_phy_file_dark,'phy_df')
#vid_file = find('*TOP1.avi',path)

root_folder = os.path.dirname(darkPath)

fps = 59.99
likelihood_threshold = 0.95
model_dt = 1/fps # Frame duration in seconds
bin_width = 20 #bin width angles
speed_threshold=0.25
ebc_angle_bin_size = 6
ebc_dist_bin_size = 10
dist_bins = 480 // ebc_dist_bin_size #480 is approximately half the arena length in pixels

pixels_per_cm_light = (dlc_df_light[dlc_df_light['top_right likelihood'] > 0.95]['top_right x'].median() - dlc_df_light[dlc_df_light['top_left likelihood'] > 0.95]['top_left x'].median()) / 60
print("P/CM LIGHT: " + str(pixels_per_cm_light))

pixels_per_cm_dark = (dlc_df_dark[dlc_df_dark['top_right likelihood'] > 0.95]['top_right x'].median() - dlc_df_dark[dlc_df_dark['top_left likelihood'] > 0.95]['top_left x'].median()) / 60
print("P/CM DARK: " + str(pixels_per_cm_dark))

head_plot_data_light, head_bin_edges_light = head_direction(dlc_df_light, phy_df_light, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_light,speed_threshold)
head_plot_data_dark, head_bin_edges_dark = head_direction(dlc_df_dark, phy_df_dark, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_dark, speed_threshold)

body_plot_data_light, body_bin_edges_light = body_direction(dlc_df_light, phy_df_light, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_light, speed_threshold)
body_plot_data_dark, body_bin_edges_dark = body_direction(dlc_df_dark, phy_df_dark, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_dark,speed_threshold)

place_cell_plots_light, x_edges_light, y_edges_light  = place_cells(dlc_df_light, phy_df_light, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_light,speed_threshold)
place_cell_plots_dark, x_edges_dark, y_edges_dark  = place_cells(dlc_df_dark, phy_df_dark, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_dark,speed_threshold)

velocity_list_light, spike_list_light, spike_avg_list_light, std_error_lower_light, std_error_upper_light, timeInSeconds_light = speedPlots(dlc_df_light, phy_df_light, dlc_phy_file_light)
velocity_list_dark, spike_list_dark, spike_avg_list_dark, std_error_lower_dark, std_error_upper_dark, timeInSeconds_dark = speedPlots(dlc_df_dark, phy_df_dark, dlc_phy_file_dark)

head_ebc_plot_data_light, head_distance_bins_light, _, _  = egocentric_head(dlc_df_light, phy_df_light, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_light,speed_threshold,ebc_angle_bin_size,ebc_dist_bin_size, dist_bins)
head_ebc_plot_data_dark, head_distance_bins_dark, _, _  = egocentric_head(dlc_df_dark, phy_df_dark, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_dark,speed_threshold,ebc_angle_bin_size,ebc_dist_bin_size, dist_bins)

body_ebc_plot_data_light, body_distance_bins_light, _, _ = egocentric_body(dlc_df_light, phy_df_light, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_light,speed_threshold,ebc_angle_bin_size,ebc_dist_bin_size, dist_bins)
body_ebc_plot_data_dark, body_distance_bins_dark, _, _ = egocentric_body(dlc_df_dark, phy_df_dark, fps, likelihood_threshold, model_dt, bin_width, dlc_phy_file_dark,speed_threshold,ebc_angle_bin_size,ebc_dist_bin_size, dist_bins)

head_ebc_plot_data_light = [row[:dist_bins] for row in head_ebc_plot_data_light]
head_distance_bins_light = head_distance_bins_light[:dist_bins]
head_ebc_plot_data_dark = [row[:dist_bins] for row in head_ebc_plot_data_dark]
head_distance_bins_dark = head_distance_bins_dark[:dist_bins]
        
body_ebc_plot_data_light = [row[:dist_bins] for row in body_ebc_plot_data_light]
body_distance_bins_light = body_distance_bins_light[:dist_bins]
body_ebc_plot_data_dark = [row[:dist_bins] for row in body_ebc_plot_data_dark]
body_distance_bins_dark = body_distance_bins_dark[:dist_bins]

with PdfPages(root_folder + "\LightVsDark.pdf") as pdf:
    
    for i in range(len(head_plot_data_light)):
        #LIGHT
        fig, axs = plt.subplots(3,4, figsize=(24, 16))
        fig.suptitle("Cell: "+ str(phy_df_light.index[i]) + " light")

        #Head_direction
        ax1 = plt.subplot(3,4,1,projection='polar')
        bin_centers = (head_bin_edges_light[:-1] + head_bin_edges_light[1:]) / 2
        bin_centers_rad = np.deg2rad(bin_centers)
        ax1.bar(bin_centers_rad, head_plot_data_light[i], width=np.deg2rad(bin_width), edgecolor='k')
        # Add labels and title
        ax1.set_title('Head Direction')
        ax1.set_theta_zero_location('N')  # Set 0 degrees to the top
        ax1.set_theta_direction(-1)  # Clockwise

        #Body_direction
        ax2 = plt.subplot(3,4,2,projection='polar')
        bin_centers = (body_bin_edges_light[:-1] + body_bin_edges_light[1:]) / 2
        bin_centers_rad = np.deg2rad(bin_centers)
        ax2.bar(bin_centers_rad, body_plot_data_light[i], width=np.deg2rad(bin_width), edgecolor='k')
        # Add labels and title
        ax2.set_title('Body Direction')
        ax2.set_theta_zero_location('N')  # Set 0 degrees to the top
        ax2.set_theta_direction(-1)  # Clockwise

        #find vmax for each head cell
        lightMax = max(max(row) for row in head_ebc_plot_data_light[i])
        darkMax  = max(max(row) for row in head_ebc_plot_data_dark[i])
        headMax  = max(lightMax, darkMax)
        
        #ebc head
        ax3 = plt.subplot(3,4,5,projection='polar')
        rbins = head_distance_bins_light.copy()
        abins = np.linspace(0,2*np.pi, (360//ebc_angle_bin_size))

        A, R = np.meshgrid(abins, rbins)


        pc = ax3.pcolormesh(A, R, head_ebc_plot_data_light[i], cmap="jet", vmin=0, vmax=headMax)
        ax3.set_theta_direction(1)
        ax3.set_theta_offset(np.pi)
        # ax3.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=['90°', '135°', '180°', '225°', '270°', '315°', '0°', '45°'])
        # ax3.set_rticks([0, 200, 400, 600, 800, 1000],labels=np.floor(np.arange(0, 1000 / pixels_per_cm_light + 1, 200/pixels_per_cm_light)))  # Less radial ticks
        ax3.axis('off')

        fig.colorbar(pc)

        #find vmax for each body cell
        lightMax = max(max(row) for row in body_ebc_plot_data_light[i])
        darkMax  = max(max(row) for row in body_ebc_plot_data_dark[i])
        bodyMax  = max(lightMax, darkMax)

        #ebc body
        ax4 = plt.subplot(3,4,6,projection='polar')
        rbins = body_distance_bins_light.copy()
        abins = np.linspace(0,2*np.pi, (360//ebc_angle_bin_size))

        A, R = np.meshgrid(abins, rbins)


        pc = ax4.pcolormesh(A, R, body_ebc_plot_data_light[i], cmap="jet", vmin=0, vmax=bodyMax)
        ax4.set_theta_direction(1)
        ax4.set_theta_offset(np.pi / 2.0)
        ax4.axis('off')
        # ax4.set_rticks([0, 200, 400, 600, 800, 1000],labels=np.floor(np.arange(0, 1000 / pixels_per_cm_light + 1, 200/pixels_per_cm_light)))  # Less radial ticks
        ax4.axis('off')

        fig.colorbar(pc)

        #find place cell limit
        place_cell_min = min(np.amax(place_cell_plots_light[i]), np.amax(place_cell_plots_dark[i]))

        #place cells
        ax5 = plt.subplot(3,4,9)
        pc = ax5.imshow(place_cell_plots_light[i], origin='lower', aspect = 'auto', extent=[x_edges_light[0], x_edges_light[-1], y_edges_light[0], y_edges_light[-1]], vmax=place_cell_min)
        fig.colorbar(pc)
        ax5.set_ylabel('Y position')
        ax5.set_title('2D Histogram of Spikes')

        #find velocity limit
        min_velocity = min(max(velocity_list_light[i]), max(velocity_list_dark[i]))

        #velocity
        ax6 = plt.subplot(3,4,10)
        ax6.plot(np.arange(0, max(velocity_list_light[i]), 5), spike_avg_list_light[i],'k')
        ax6.fill_between(np.arange(0, max(velocity_list_light[i]), 5), std_error_lower_light[i], std_error_upper_light[i],color='#ADD8E6')
        ax6.set_xlabel("Velocity")
        ax6.set_ylabel("Firing Rate")
        ax6.set_ylabel("Firing Rate vs Velocity")
        ax6.set_xlim(xmax=min_velocity)
        ax6.set_ylim(bottom=0)

        #DARK
        # fig, ((ax1,ax2),(ax3,ax4), (ax5, ax6))= plt.subplots(3,2, figsize=(12, 12))
        fig.suptitle("Cell: "+ str(phy_df_dark.index[i]) + " Light/Dark")

        #Head_direction
        ax1 = plt.subplot(3,4,3,projection='polar')
        bin_centers = (head_bin_edges_dark[:-1] + head_bin_edges_dark[1:]) / 2
        bin_centers_rad = np.deg2rad(bin_centers)
        ax1.bar(bin_centers_rad, head_plot_data_dark[i], width=np.deg2rad(bin_width), edgecolor='k')
        # Add labels and title
        ax1.set_title('Head Direction')
        ax1.set_theta_zero_location('N')  # Set 0 degrees to the top
        ax1.set_theta_direction(-1)  # Clockwise

        #Body_direction
        ax2 = plt.subplot(3,4,4,projection='polar')
        bin_centers = (body_bin_edges_dark[:-1] + body_bin_edges_dark[1:]) / 2
        bin_centers_rad = np.deg2rad(bin_centers)
        ax2.bar(bin_centers_rad, body_plot_data_dark[i], width=np.deg2rad(bin_width), edgecolor='k')
        # Add labels and title
        ax2.set_title('Body Direction')
        ax2.set_theta_zero_location('N')  # Set 0 degrees to the top
        ax2.set_theta_direction(-1)  # Clockwise

        #ebc head
        ax3 = plt.subplot(3,4,7,projection='polar')
        rbins = head_distance_bins_dark.copy()
        abins = np.linspace(0,2*np.pi, (360//ebc_angle_bin_size))

        A, R = np.meshgrid(abins, rbins)


        pc = ax3.pcolormesh(A, R, head_ebc_plot_data_dark[i], cmap="jet", vmin=0, vmax=headMax)
        ax3.set_theta_direction(1)
        ax3.set_theta_offset(np.pi)
        ax3.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=['90°', '135°', '180°', '225°', '270°', '315°', '0°', '45°'])
        # ax3.set_rticks([0, 200, 400, 600, 800, 1000],labels=np.floor(np.arange(0, 1000 / pixels_per_cm_dark + 1, 200/pixels_per_cm_dark)))  # Less radial ticks
        ax3.axis('off')
        fig.colorbar(pc)

        #ebc body
        ax4 = plt.subplot(3,4,8,projection='polar')
        rbins = body_distance_bins_dark.copy()
        abins = np.linspace(0,2*np.pi, (360//ebc_angle_bin_size))

        A, R = np.meshgrid(abins, rbins)


        pc = ax4.pcolormesh(A, R, body_ebc_plot_data_dark[i], cmap="jet", vmin = 0, vmax=bodyMax)
        ax4.set_theta_direction(1)
        ax4.set_theta_offset(np.pi / 2.0)
        # ax4.set_rticks([0, 200, 400, 600, 800, 1000],labels=np.floor(np.arange(0, 1000 / pixels_per_cm_dark + 1, 200/pixels_per_cm_dark)))  # Less radial ticks
        ax4.axis('off')
        fig.colorbar(pc)

        #place cells
        ax5 = plt.subplot(3,4,11)
        pc = ax5.imshow(place_cell_plots_dark[i], origin='lower', aspect = 'auto', extent=[x_edges_dark[0], x_edges_dark[-1], y_edges_dark[0], y_edges_dark[-1]], vmax=place_cell_min)
        fig.colorbar(pc)
        ax5.set_ylabel('Y position')
        ax5.set_title('2D Histogram of Spikes')

        #velocity
        ax6 = plt.subplot(3,4,12)
        ax6.plot(np.arange(0, max(velocity_list_dark[i]), 5), spike_avg_list_dark[i],'k')
        ax6.fill_between(np.arange(0, max(velocity_list_dark[i]), 5), std_error_lower_dark[i], std_error_upper_dark[i],color='#ADD8E6')
        ax6.set_xlabel("Velocity")
        ax6.set_ylabel("Firing Rate")
        ax6.set_title("Firing Rate vs Velocity")
        ax6.set_xlim(xmax=min_velocity)
        ax6.set_ylim(bottom=0)

        fig.tight_layout()

        pdf.savefig(fig)
