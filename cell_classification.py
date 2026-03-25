import numpy as np

def classify_cell(
    dlc_df,
    phy_df,
    MRLS_1,
    MRLS_2,
    MALS_1,
    MALS_2,
    pref_dist_1,
    pref_dist_2,
    Mrlthresh_1,
    Mrlthresh_2,
    debug: bool = True,
    debug_first_n: int = 10,
):
    """
    Classify each cell based on egocentric boundary cell (EBC) criteria using head- and body-centered data.

    Args:
        dlc_df (DataFrame): DataFrame with body coordinates and timestamps.
        phy_df (DataFrame): DataFrame with spike time data for each cell.
        MRLs_h, Mrlthresh_h, MALs_h, pref_dist_head (arrays): Head-centered metrics for MRL, threshold, angles, and preferred distances.
        MRLS_1_h, MRLS_2_h, MALS_1_h, MALS_2_h (arrays): Head-centered data for first and second halves of session.
        MRLs_b, Mrlthresh_b, MALs_b, pref_dist_body (arrays): Body-centered metrics for MRL, threshold, angles, and preferred distances.
        MRLS_1, MRLS_2, MALS_1, MALS_2 (arrays): Body-centered data for first and second halves of session.
        pref_dist_1, pref_dist_2, pref_dist_1_h, pref_dist_2_h (arrays): Preferred distances for body and head for first and second halves.

    Returns:
        list: Cell types classified as 'ebc' (egocentric boundary cell) or 'not ebc'.
    """
    fps = 59.99
    cell_types = []
    debug_printed = 0
    rec_duration = len(dlc_df)/fps
    for i in range(len(phy_df)):
        cell = phy_df.iloc[i]
        firing_rate =  len(cell.spikeT)/rec_duration 

        # Condition 1: Cell firing rate must exceed threshold of 0.1 Hz
        # removing this condition for now since some cells have very low firing rates but still show strong tuning, can add back in later if needed PRLP 03/20/2026
        # condition_1 = firing_rate > 0.1
            
        # Condition 2: MRL in both halves exceeds (half-specific) thresholds
        condition_2_1 = MRLS_1[i] > Mrlthresh_1[i]
        condition_2_2 = MRLS_2[i] > Mrlthresh_2[i]
        condition_2 = condition_2_1 and condition_2_2

        # (3) The change in Mean Resultant Angle (MRA) between the 1st and 2nd half was <45°
        # Assuming MALS_1 and MALS_2 are given in radians.
        # Use circular difference on angles (wrap-aware) so e.g. +179 and -179 => ~2°, not ~358°.
        mra1_deg = np.degrees(MALS_1[i])
        mra2_deg = np.degrees(MALS_2[i])
        delta_deg = (mra1_deg - mra2_deg + 180.0) % 360.0 - 180.0
        change_in_MRA = np.abs(delta_deg)
        condition_3 = change_in_MRA < 45

        # (4) The change in preferred boundary distance between the 1st and 2nd half was <75% of the preferred distance for the whole session
        change_in_pref_dist = np.abs(pref_dist_1[i] - pref_dist_2[i])
        condition_4 = change_in_pref_dist < 0.75 * np.mean([pref_dist_1[i], pref_dist_2[i]])

        # Final checks
        # print(i, condition_1,condition_2,condition_3,condition_4)
        if condition_2 and condition_3 and condition_4: #condition_1 and PRLP 03/20/2026
            cell_types.append('ebc')
        else:
            if debug and debug_printed < debug_first_n:
                failed = []
                if not condition_2:
                    failed.append(
                        f"cond2 (half1 {MRLS_1[i]:.3f}>{Mrlthresh_1[i]:.3f}, half2 {MRLS_2[i]:.3f}>{Mrlthresh_2[i]:.3f})"
                    )
                if not condition_3:
                    failed.append(f"cond3 (MRA delta={change_in_MRA:.1f} deg)")
                if not condition_4:
                    failed.append(f"cond4 (pref_dist delta={change_in_pref_dist:.1f})")
                print(f"Cell {phy_df.index[i]} NOT EBC: failed {'; '.join(failed)}")
                debug_printed += 1
            cell_types.append('not ebc')
    
    return cell_types

# def classify_cell(dlc_df,phy_df,MRLs_h,Mrlthresh_h,MALs_h,pref_dist_head, MRLS_1_h,MRLS_2_h,MALS_1_h,MALS_2_h,MRLs_b,Mrlthresh_b,MALs_b,pref_dist_body,MRLS_1_b, MRLS_2_b, MALS_1_b, MALS_2_b,pref_dist_1_b,pref_dist_2_b,pref_dist_1_h,pref_dist_2_h):
#     """
#     Classify each cell based on egocentric boundary cell (EBC) criteria using head- and body-centered data.

#     Args:
#         dlc_df (DataFrame): DataFrame with body coordinates and timestamps.
#         phy_df (DataFrame): DataFrame with spike time data for each cell.
#         MRLs_h, Mrlthresh_h, MALs_h, pref_dist_head (arrays): Head-centered metrics for MRL, threshold, angles, and preferred distances.
#         MRLS_1_h, MRLS_2_h, MALS_1_h, MALS_2_h (arrays): Head-centered data for first and second halves of session.
#         MRLs_b, Mrlthresh_b, MALs_b, pref_dist_body (arrays): Body-centered metrics for MRL, threshold, angles, and preferred distances.
#         MRLS_1, MRLS_2, MALS_1, MALS_2 (arrays): Body-centered data for first and second halves of session.
#         pref_dist_1, pref_dist_2, pref_dist_1_h, pref_dist_2_h (arrays): Preferred distances for body and head for first and second halves.

#     Returns:
#         list: Cell types classified as 'ebc' (egocentric boundary cell) or 'not ebc'.
#     """
#     fps = 59.99
#     cell_numbers = phy_df.index
#     cell_type = []
#     rec_duration = len(dlc_df)/fps
#     for i in range(len(cell_numbers)):
#         cell = phy_df.iloc[i]
#         firing_rate =  len(cell.spikeT)/rec_duration 

#         # Condition 1: Cell firing rate must exceed threshold of 0.1 Hz
#         condition_1 = firing_rate > 0.1
        
#         # Condition 2: MRL in both halves exceeds threshold
#         condition_2 = (MRLS_1[i] > Mrlthresh_b[i]) and (MRLS_2[i] > Mrlthresh_b[i])

#         # (3) The change in Mean Resultant Angle (MRA) between the 1st and 2nd half was <45°
#         # Assuming MALS_1 and MALS_2 are given in radians, convert to degrees if necessary
#         change_in_MRA = np.abs(np.degrees(MALS_1[i] - MALS_2[i]))
#         condition_3 = change_in_MRA < 45

#         # (4) The change in preferred boundary distance between the 1st and 2nd half was <75% of the preferred distance for the whole session
#         change_in_pref_dist = np.abs(pref_dist_1[i] - pref_dist_2[i])
#         condition_4 = change_in_pref_dist < 0.75 * np.mean([pref_dist_1[i], pref_dist_2[i]])

#         # Final checks
#         print(i, condition_1,condition_2,condition_3,condition_4)
#         if condition_1 and condition_2 and condition_3 and condition_4:  # was missing condition 4 before 7/22/2025
           
#             cell_type.append('ebc')
#         else:
#             cell_type.append('not ebc')
    
#     return cell_type