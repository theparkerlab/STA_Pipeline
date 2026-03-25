# Imports
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
import h5py
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Custom imports from specific modules
from speed import speedPlots
from body_direction import body_direction
from Egocentric import *
from Head_direction import *
from egocentric_head import *
from place_cells import *
from utils import *
from trajectory_head import *
from trajectory_body import *
from bootstrap_ebc_head import *
from bootstrap_ebc_body import *
from half_check import *
from half_check_body import *
from cell_classification import *
from significance_plots import create_significance_plots

# TEST MODE: set to True to process only the first N cells and save outputs
# into a "test" subdirectory (for faster trial runs).
TEST_MODE = True   # True = first 10 cells, save in test/; False = all cells, normal paths
MAX_TEST_CELLS = 10

# File-finding function
def find(pattern, path):
    """
    Function to search for files matching a given pattern within a specified path.
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files: 
            if fnmatch.fnmatch(name, pattern): 
                result.append(os.path.join(root, name))
    if len(result) == 1:
        result = result[0]
    return result

#function to save analyzed data to an h5 file
def save_to_h5(filename, **variables):
    with h5py.File(filename, 'w') as f:
        for name, obj in variables.items():
            if isinstance(obj, np.ndarray):
                f.create_dataset(name, data=obj,
                                 compression='gzip', chunks=True)
            elif isinstance(obj, list):
                # convert to array (must be homogenous dtype)
                arr = np.array(obj)
                f.create_dataset(name, data=arr,
                                 compression='gzip', chunks=True)
            elif isinstance(obj, pd.DataFrame):
                grp = f.create_group(name)
                # values
                grp.create_dataset('values', data=obj.values,
                                   compression='gzip', chunks=True)
                # column & index labels as fixed-length ASCII
                dt = h5py.string_dtype(encoding='utf-8')
                grp.create_dataset('columns',
                                   data=np.array(obj.columns.astype(str), dtype=dt))
                grp.create_dataset('index',
                                   data=np.array(obj.index.astype(str), dtype=dt))
            else:
                # fallback: pickle anything else into an HDF5 attribute
                import pickle
                f.attrs[name] = np.void(pickle.dumps(obj))


# File/Video Initialization
root = Tk()
root.withdraw() 
path = askdirectory(title='Choose experiment folder', initialdir=r'\\rhea\E\ephys')
print('you have selected: ', path)

# Locate the file for data analysis
dlc_phy_file = find('*topDLCephys.h5', path)

# Load data into dataframes
print('loading DLC and ephys data...')
dlc_df = pd.read_hdf(dlc_phy_file, 'dlc_df')
phy_df = pd.read_hdf(dlc_phy_file, 'phy_df')
print('done loading DLC and ephys data.')

# Test mode: limit to first N cells and send all outputs to a "test" subdirectory
if TEST_MODE:
    phy_df = phy_df.iloc[:MAX_TEST_CELLS].copy()
    test_dir = os.path.join(os.path.dirname(dlc_phy_file), 'test')
    os.makedirs(test_dir, exist_ok=True)
    output_base = os.path.join(test_dir, os.path.basename(dlc_phy_file)[:-3])  # path prefix for saved files
    save_file = output_base + '.h5'  # passed to modules that build paths from file[:-3]
    print(f'TEST MODE: processing first {len(phy_df)} cells; outputs -> {test_dir}')
else:
    output_base = dlc_phy_file[:-3]
    save_file = dlc_phy_file

# Set parameters for analysis
fps = 59.99
likelihood_threshold = 0.95
model_dt = 1 / fps  # Frame duration in seconds
bin_width = 20  # Bin width for angles
speed_threshold = 0.25
ebc_angle_bin_size = 6
ebc_dist_bin_size = 10
dist_bins = 480 // ebc_dist_bin_size #480 is approximately half the arena length in pixels
pixels_per_cm = (dlc_df[dlc_df['top_right likelihood'] > 0.95]['top_right x'].median() - dlc_df[dlc_df['top_left likelihood'] > 0.95]['top_left x'].median()) / 60
print("P/CM: " + str(pixels_per_cm))

# Run analyses for head and body direction, and place cells
print('analyzing head direction...')
head_plot_data, head_bin_edges = head_direction(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold)
print('analyzing movement direction...')
body_plot_data, body_bin_edges = body_direction(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold)
print('analyzing allocentric...')
place_cell_plots, x_edges, y_edges = place_cells(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold)
print('analyzing velocity...')
velocity_list, spike_list, spike_avg_list, std_error_lower, std_error_upper, timeInSeconds = speedPlots(dlc_df, phy_df, save_file)

# Run trajectory analyses for head and body
print('getting head trajectories...')
mouse_xs, mouse_ys, cell_hds, ch_points_x, ch_points_y = trajectory_head(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold)
print('getting body trajectories...')
mouse_xsb, mouse_ysb, cell_bds, ch_points_x, ch_points_y = trajectory_body(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold)

# Bootstrap analyses for egocentric head and body
print('analyzing egocentric head...')
MRLs_h, Mrlthresh_h, MALs_h, head_ebc_plot_data, head_distance_bins, ebc_plot_data_binary_head, max_bins_head, pref_dist_head, shuffled_mrls_head, Mrlthresh_half1_h, Mrlthresh_half2_h = bootstrap_egocentric_head(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size, dist_bins)
print()
print('\nHead MRLs:')
for i, mrl in enumerate(MRLs_h):
    print(f'  Cell {i}: MRL = {mrl:.4f}')

print('analyzing egocentric movement direction...')
MRLs_b, Mrlthresh_b, MALs_b, body_ebc_plot_data, body_distance_bins, ebc_plot_data_binary, max_bins, pref_dist_body, shuffled_mrls_body, Mrlthresh_half1_b, Mrlthresh_half2_b = bootstrap_egocentric_body(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size, dist_bins)


print('\nBody MRLs:')
for i, mrl in enumerate(MRLs_b):
    print(f'  Cell {i}: MRL = {mrl:.4f}')

## DOUBLE check that this is working correctly (only looking at first vs. second half of dataset)
## right now it's splitting the DLC data in half in half_check.py, are the correct time values in place and are the spike times being correctly checked against the DLC times?
## do we have to also split the spike times in half and feed those in...
# Half-check for consistency in analysis
print('first vs. second half egocentric head...')
half_check_file = output_base + '_half_ebc_head_data'
MRLS_1_h, MRLS_2_h, MALS_1_h, MALS_2_h, pref_dist_1_h, pref_dist_2_h, half_ebc_head_1, half_ebc_head_2 = egocentric_head_half_check(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size, dist_bins)

print('first vs. second half egocentric movement direction...')
half_check_file = output_base + '_half_ebc_body_data'
MRLS_1_b, MRLS_2_b, MALS_1_b, MALS_2_b, pref_dist_1_b, pref_dist_2_b, half_ebc_body_1, half_ebc_body_2 = egocentric_body_half_check(dlc_df, phy_df, fps, likelihood_threshold, model_dt, bin_width, save_file, speed_threshold, ebc_angle_bin_size, ebc_dist_bin_size, dist_bins)

#print(len(MRLS_1), len(MRLS_2), len(MALS_1), len(MALS_2))

# Classify cell types
print('classifying cell types...')
# cell_type = classify_cell(dlc_df, phy_df, MRLs_h, Mrlthresh_h, MALs_h, pref_dist_head, MRLS_1_h, MRLS_2_h, MALS_1_h, MALS_2_h, MRLs_b, Mrlthresh_b, MALs_b, pref_dist_body, MRLS_1_b, MRLS_2_b, MALS_1_b, MALS_2_b, pref_dist_1_b, pref_dist_2_b, pref_dist_1_h, pref_dist_2_h)

ref_frames, cell_types, MRLS_1, MRLS_2, MALS_1, MALS_2, pref_dist_1, pref_dist_2, Mrlthresh, Mrlthresh_half1, Mrlthresh_half2, full_session_MRL = ([] for i in range(12))

for i in range(len(phy_df)):
    head_MRL = MRLs_h[i]
    body_MRL = MRLs_b[i]
    if head_MRL > body_MRL:
        MRLS_1.append(MRLS_1_h[i])
        MRLS_2.append(MRLS_2_h[i])
        MALS_1.append(MALS_1_h[i])
        MALS_2.append(MALS_2_h[i])
        pref_dist_1.append(pref_dist_1_h[i])
        pref_dist_2.append(pref_dist_2_h[i])
        Mrlthresh.append(Mrlthresh_h[i])
        Mrlthresh_half1.append(Mrlthresh_half1_h[i])
        Mrlthresh_half2.append(Mrlthresh_half2_h[i])
        full_session_MRL.append(MRLs_h[i])
        ref_frames.append('head')
    else:
        MRLS_1.append(MRLS_1_b[i])
        MRLS_2.append(MRLS_2_b[i])
        MALS_1.append(MALS_1_b[i])
        MALS_2.append(MALS_2_b[i])
        pref_dist_1.append(pref_dist_1_b[i])
        pref_dist_2.append(pref_dist_2_b[i])
        Mrlthresh.append(Mrlthresh_b[i])
        Mrlthresh_half1.append(Mrlthresh_half1_b[i])
        Mrlthresh_half2.append(Mrlthresh_half2_b[i])
        full_session_MRL.append(MRLs_b[i])
        ref_frames.append('body')

cell_types = classify_cell(
    dlc_df,
    phy_df,
    MRLS_1,
    MRLS_2,
    MALS_1,
    MALS_2,
    pref_dist_1,
    pref_dist_2,
    Mrlthresh_half1,
    Mrlthresh_half2,
)

# Round analysis results for readability
MRLs_h = [round(num, 3) for num in MRLs_h]
Mrlthresh_h = [round(num, 3) for num in Mrlthresh_h]
MALs_h = [round(num, 3) for num in MALs_h]
MRLs_b = [round(num, 3) for num in MRLs_b]
Mrlthresh_b = [round(num, 3) for num in Mrlthresh_b]
MALs_b = [round(num, 3) for num in MALs_b]
MRLS_1_b = [round(num, 3) for num in MRLS_1_b]
MRLS_2_b = [round(num, 3) for num in MRLS_2_b]
MALS_1_b = [round(num, 3) for num in MALS_1_b]
MALS_2_b = [round(num, 3) for num in MALS_2_b]
MRLS_1_h = [round(num, 3) for num in MRLS_1_h]
MRLS_2_h = [round(num, 3) for num in MRLS_2_h]
MALS_1_h = [round(num, 3) for num in MALS_1_h]
MALS_2_h = [round(num, 3) for num in MALS_2_h]

#print(Mrlthresh_h)

# Define the PDF filename for saving plots
filename = output_base + "angle_" + str(ebc_angle_bin_size) + "dis_" + str(ebc_dist_bin_size) + "_summary_plots.pdf"

#11/10/24 ask joy or krithik to modify these data that go into these plots to exclude the outer zone showing up as artifact (have to do this in the original calculation before MRL/MRA etc. are done)
# Save plots to PDF
print('creating final PDF with plots...')
with PdfPages(filename) as pdf:
    for i in tqdm(range(len(head_plot_data))):
        # Create figure and subplots with enough spacing to avoid caption/label overlap
        fig = plt.figure(figsize=(10.8, 9.5))
        fig.suptitle(f"Cell: {phy_df.index[i]} | {cell_types[i]} | ref: {ref_frames[i]}")
        gs = fig.add_gridspec(4, 2, hspace=0.58, wspace=0.48)

        # Plot head direction in polar coordinates
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        bin_centers = (head_bin_edges[:-1] + head_bin_edges[1:]) / 2
        bin_centers_rad = np.deg2rad(bin_centers)
        ax1.bar(bin_centers_rad, head_plot_data[i], width=np.deg2rad(bin_width), edgecolor='k')
        ax1.set_title('Head Direction', pad=8)
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_frame_on(False)
        ax1.tick_params(pad=3)  # pull angle labels in to avoid overlap with row below

        # Plot body direction in polar coordinates
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        bin_centers = (body_bin_edges[:-1] + body_bin_edges[1:]) / 2
        bin_centers_rad = np.deg2rad(bin_centers)
        ax2.bar(bin_centers_rad, body_plot_data[i], width=np.deg2rad(bin_width), edgecolor='k')
        ax2.set_title('Body Direction', pad=8)
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_frame_on(False)
        ax2.tick_params(pad=3)  # pull angle labels in to avoid overlap with row below

        # EBC head ratemap (directly under head direction)
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        rbins = head_distance_bins.copy()
        abins = np.linspace(0, 2 * np.pi, (360 // ebc_angle_bin_size))
        A, R = np.meshgrid(abins, rbins)
        ax3.grid(False)
        pc = ax3.pcolormesh(A, R, head_ebc_plot_data[i], cmap="jet", edgecolors="none", rasterized=True)
        ax3.set_theta_direction(1)
        ax3.set_theta_offset(np.pi)
        # ax3.set_rticks([0, 200, 400, 600, 800, 1000], labels=np.floor(np.arange(0, 1000 / pixels_per_cm + 1, 200 / pixels_per_cm)))
        ax3.set_title(f"MRL: {MRLs_h[i]:.3f}  MRA: {MALs_h[i]:.3f}  pref_dist: {pref_dist_head[i]:.3f}", fontsize=8, pad=6)
        ax3.axis('off')
        ax3.set_frame_on(False)
        fig.colorbar(pc)

        # EBC body ratemap (directly under body direction)
        ax4 = fig.add_subplot(gs[1, 1], projection='polar')
        rbins = body_distance_bins.copy()
        abins = np.linspace(0, 2 * np.pi, (360 // ebc_angle_bin_size))
        A, R = np.meshgrid(abins, rbins)
        ax4.grid(False)
        pc = ax4.pcolormesh(A, R, body_ebc_plot_data[i], cmap="jet", edgecolors="none", rasterized=True)
        ax4.set_theta_direction(1)
        ax4.set_theta_offset(np.pi / 2.0)
        # ax4.set_rticks([0, 200, 400, 600, 800, 1000], labels=np.floor(np.arange(0, 1000 / pixels_per_cm + 1, 200 / pixels_per_cm)))
        ax4.set_title(f"MRL: {MRLs_b[i]:.3f}  MRA: {MALs_b[i]:.3f}  pref_dist: {pref_dist_body[i]:.3f}", fontsize=8, pad=6)
        ax4.axis('off')
        ax4.set_frame_on(False)
        fig.colorbar(pc)

        # Place cell plot (Gaussian-smoothed 2D histogram)
        filt_size = 3
        place_smoothed = cv2.GaussianBlur(
            np.nan_to_num(place_cell_plots[i], nan=0.0).astype(np.float32),
            (filt_size, filt_size), filt_size
        )
        ax5 = fig.add_subplot(gs[2, 0])
        pc = ax5.imshow(place_smoothed, origin='lower', aspect='auto', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        fig.colorbar(pc)
        ax5.set_ylabel('Y position')
        ax5.set_title('2D Histogram of Spikes')
        ax5.axis("square")
        ax5.spines[['top', 'right']].set_visible(False)

        # Firing rate vs velocity plot (y-axis fits this cell's data)
        ax6 = fig.add_subplot(gs[2, 1])
        x_vel = np.arange(0, max(velocity_list[i]), 5)
        ax6.plot(x_vel, spike_avg_list[i], 'k')
        ax6.fill_between(x_vel, std_error_lower[i], std_error_upper[i], color='#ADD8E6')
        ax6.set_title('Firing Rate vs Velocity')
        ax6.set_xlabel("Velocity")
        ax6.set_ylabel("Firing Rate")
        y_max = max(np.nanmax(std_error_upper[i]) if len(std_error_upper[i]) else 0, np.nanmax(spike_avg_list[i]) if len(spike_avg_list[i]) else 0, 0.0)
        ax6.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
        ax6.spines[['top', 'right']].set_visible(False)

        # Mouse position with head direction color coding
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(ch_points_x, ch_points_y, ".", markersize=0.5, color='lightgrey', alpha=0.3, rasterized=True)
        scatter = ax7.scatter(mouse_xs[i], mouse_ys[i], c=cell_hds[i], cmap='hsv', vmin=0, vmax=360, s=5, alpha=1.0, zorder=10, rasterized=True)
        cbar = plt.colorbar(scatter, ax=ax7, shrink=0.8, pad=0.02)
        cbar.set_label('Head Dir. (deg)', fontsize=8)
        ax7.set_xlabel('Mouse X Position')
        ax7.set_ylabel('Mouse Y Position')
        ax7.set_title('Mouse Position with Color-Coded Head Direction', fontsize=9)
        ax7.axis("square")
        ax7.spines[['top', 'right']].set_visible(False)

        # Mouse position with body direction color coding
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.plot(ch_points_x, ch_points_y, ".", markersize=0.5, color='lightgrey', alpha=0.3, rasterized=True)
        scatter2 = ax8.scatter(mouse_xsb[i], mouse_ysb[i], c=cell_bds[i], cmap='hsv', vmin=0, vmax=360, s=5, alpha=1.0, zorder=10, rasterized=True)
        cbar = plt.colorbar(scatter2, ax=ax8, shrink=0.8, pad=0.02)
        cbar.set_label('Body Dir. (deg)', fontsize=8)
        ax8.set_xlabel('Mouse X Position')
        ax8.set_ylabel('Mouse Y Position')
        ax8.set_title('Mouse Position with Color-Coded Body Direction', fontsize=9)
        ax8.axis("square")
        ax8.spines[['top', 'right']].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96], pad=1.0, w_pad=1.2, h_pad=1.2)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print('done creating PDF.')

# Significance plots (bootstrap, first/second half, full-session summary)
create_significance_plots(
    phy_df, ref_frames, cell_types,
    MRLs_h, Mrlthresh_h, MALs_h,
    MRLs_b, Mrlthresh_b, MALs_b,
    head_ebc_plot_data, head_distance_bins,
    body_ebc_plot_data, body_distance_bins,
    ebc_plot_data_binary_head, ebc_plot_data_binary,
    max_bins_head, max_bins,
    pref_dist_head, pref_dist_body,
    MRLS_1, MRLS_2, MALS_1, MALS_2, pref_dist_1, pref_dist_2,
    half_ebc_head_1, half_ebc_head_2,
    half_ebc_body_1, half_ebc_body_2,
    shuffled_mrls_head, shuffled_mrls_body,
    full_session_MRL, Mrlthresh,
    ebc_angle_bin_size, ebc_dist_bin_size,
    output_base,
)
print('done creating significance plots.')

print('saving variables to joblib file...')
names = ['cell_types','ref_frames','head_plot_data','head_bin_edges','body_plot_data','body_bin_edges','place_cell_plots','x_edges','y_edges','velocity_list','spike_list','spike_avg_list',
         'std_error_lower','std_error_upper','timeInSeconds','mouse_xs','mouse_ys','cell_hds','ch_points_x','ch_points_y','mouse_xsb','mouse_ysb','cell_bds',
         'ch_points_x','ch_points_y','MRLs_h','Mrlthresh_h','MALs_h','head_ebc_plot_data','head_distance_bins','ebc_plot_data_binary_head','max_bins_head',
         'pref_dist_head','MRLs_b','Mrlthresh_b','MALs_b','body_ebc_plot_data','body_distance_bins','ebc_plot_data_binary','max_bins','pref_dist_body',
         'MRLS_1_b','MRLS_2_b','MALS_1_b','MALS_2_b','pref_dist_1_b','pref_dist_2_b','MRLS_1_h','MRLS_2_h','MALS_1_h','MALS_2_h','pref_dist_1_h','pref_dist_2_h']

to_save = {n: globals()[n] for n in names}
joblib_name = output_base + "_analyzed.joblib"
joblib.dump(to_save, joblib_name, compress=('lz4', 3))
# …later…
#loaded = joblib.load('all_data.joblib')
print('done saving - analysis complete!')