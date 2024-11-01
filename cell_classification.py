import numpy as np


def classify_cell(dlc_df,phy_df,MRLs_h,Mrlthresh_h,MALs_h,pref_dist_head, MRLS_1_h,MRLS_2_h,MALS_1_h,MALS_2_h,MRLs_b,Mrlthresh_b,MALs_b,pref_dist_body,MRLS_1, MRLS_2, MALS_1, MALS_2,pref_dist_1,pref_dist_2,pref_dist_1_h,pref_dist_2_h):
    fps = 59.99
    cell_numbers = phy_df.index
    cell_type = []
    rec_duration = len(dlc_df)/fps
    for i in range(len(cell_numbers)):
        cell = phy_df.iloc[i]
        firing_rate =  len(cell.spikeT)/rec_duration 
        condition_1 = firing_rate > 0.1
        
        condition_2 = (MRLS_1[i] > Mrlthresh_b[i]) and (MRLS_2[i] > Mrlthresh_b[i])

        # (3) The change in Mean Resultant Angle (MRA) between the 1st and 2nd half was <45°
        # Assuming MALS_1 and MALS_2 are given in radians, convert to degrees if necessary
        change_in_MRA = np.abs(np.degrees(MALS_1[i] - MALS_2[i]))
        condition_3 = change_in_MRA < 45

        # (4) The change in preferred boundary distance between the 1st and 2nd half was <75% of the preferred distance for the whole session
        change_in_pref_dist = np.abs(pref_dist_1[i] - pref_dist_2[i])
        condition_4 = change_in_pref_dist < 0.75 * np.mean([pref_dist_1[i], pref_dist_2[i]])

        # Final checks
        print(i, condition_1,condition_2,condition_3,condition_4)
        if condition_1 and condition_2 and condition_3:
           
            cell_type.append('ebc')
        else:
            cell_type.append('not ebc')
    
    return cell_type