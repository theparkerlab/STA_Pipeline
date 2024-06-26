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


#File/Video Initialization
# root = Tk()
# root.withdraw() 
# path = askdirectory(title='Choose experiment folder', initialdir=r'\\rhea\D\ephys') # show an "Open" dialog box and return the path to the selected file
# print('you have selected: ', path)

# dlc_phy_file = find('*topDLCephys.h5',path)
dlc_df = pd.read_hdf(r"\\rhea\D\ephys\20240608\WT0008LN\FM\20240608_WT0008LN_M_FM_DMS1_angie_ephys_topDLCephys.h5", 'dlc_df')
phy_df = pd.read_hdf(r"\\rhea\D\ephys\20240608\WT0008LN\FM\20240608_WT0008LN_M_FM_DMS1_angie_ephys_topDLCephys.h5",'phy_df')
#vid_file = find('*TOP1.avi',path)

fps = 59.99
likelihood_threshold = 0.95
model_dt = 1/fps # Frame duration in seconds
bin_width = 20 #bin width angles

'''
# Adding timestamps to dlc file and only considering columns of interest
dlc_df['time'] = np.arange(len(dlc_df))/fps

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
'''

speedPlots(dlc_df, phy_df)