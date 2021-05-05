import numpy as np
import time

def make_dFoF(F , Fneu , neuropil_fraction , percentile_baseline):
    """
    calculates the dF/F and other signals. Designed for Suite2p data.
    See S2p documentation for more details
    RH 2021
    Args:
        F (Path): raw fluorescence values of each ROI. dims(time, ...)
        Fneu (np.ndarray): neuropil signals corresponding to each ROI. dims match F.
        neuropil_fraction (float): value, 0-1, of neuropil signal (Fneu) to be subtracted off of ROI signals (F)
        percentile_baseline (float/int): value, 0-100, of percentile to be subtracted off from signals
    Returns:
        positions_new_sansOutliers (np.ndarray): positions
        positions_new_absolute_sansOutliers (np.ndarray): absolute positions
    """
    tic = time.time()

    F_neuSub = F - neuropil_fraction*Fneu
    F_baseline = np.percentile(F_neuSub , percentile_baseline , axis=0)
    dF = F_neuSub - F_baseline
    dFoF = dF / F_baseline
    print(f'Calculated dFoF. Total elapsed time: {round(time.time() - tic,2)} seconds')
    
    return dFoF , dF , F_neuSub , F_baseline

