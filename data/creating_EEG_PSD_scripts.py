from scipy.io import loadmat, savemat
from scipy.signal import welch
import numpy as np
import os 
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Classes.config import Config


def creating_EEG_PSD_scripts():
    # ----------------------------
    # Parameters
    # ----------------------------
    FS = 128.0  # Hz

    # Frequency bands (standard EEG)
    THETA_BAND = (4, 8)
    ALPHA_BAND = (8, 13)
    BETA_BAND  = (13, 30)

    # ----------------------------
    # Load original dataset.mat
    # ----------------------------
    dataset = loadmat(Config.data_dir)

    EEGsample = dataset["EEGsample"]  # (2022, 30, 384)
    substate  = dataset["substate"]   # (2022, 1)

    n_trials, n_channels, n_samples = EEGsample.shape
    print("EEGsample shape:", EEGsample.shape)

    # ----------------------------
    # Helper: compute band powers for one 1D signal
    # ----------------------------
    def compute_band_powers(signal_1d, fs=FS):
        """
        Compute alpha, beta, theta band power and (alpha+theta)/beta
        using Welch PSD.
        """
        freqs, psd = welch(signal_1d, fs=fs, nperseg=len(signal_1d))

        theta_mask = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
        alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
        beta_mask  = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])

        theta_power = psd[theta_mask].sum()
        alpha_power = psd[alpha_mask].sum()
        beta_power  = psd[beta_mask].sum()

        if beta_power > 0:
            ratio = (alpha_power + theta_power) / beta_power
        else:
            ratio = 0.0

        # Return in the order you specified: alpha, beta, theta, ratio
        return alpha_power, beta_power, theta_power, ratio

    # ----------------------------
    # Allocate feature matrix
    # ----------------------------
    feature_matrix = np.zeros((n_trials, n_channels, 4), dtype=float)

    # ----------------------------
    # Compute features
    # ----------------------------
    for t in range(n_trials):
        for ch in range(n_channels):
            sig = EEGsample[t, ch, :]  # (384,)
            alpha, beta, theta, ratio = compute_band_powers(sig, fs=FS)
            feature_matrix[t, ch, 0] = alpha
            feature_matrix[t, ch, 1] = beta
            feature_matrix[t, ch, 2] = theta
            feature_matrix[t, ch, 3] = ratio

    print("feature_matrix shape:", feature_matrix.shape)  # (2022, 30, 4)

    # ----------------------------
    # Labels
    # ----------------------------
    labels = substate.copy()  # (2022, 1)

    # ----------------------------
    # Save to EEG_PSD_Features.mat
    # ----------------------------
    savemat(Config.psd_dir, {
        "feature_matrix": feature_matrix,
        "labels": labels
    })

    print("Saved EEG_PSD_Features.mat")
    
