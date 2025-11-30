import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
import sys 
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Classes.config import Config

def csp_eda():
    # Load data
    data = loadmat(Config.data_dir)
    X = data["EEGsample"]  # Shape (2022, 30, 384)
    X = X.astype(np.float64)
    y = data["substate"].ravel()  # Flatten labels

    # Shuffle data
    np.random.seed(0)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Apply CSP with all components (30)
    n_components = Config.num_components
    csp = CSP(n_components=n_components, log=True)
    csp.fit(X, y)
    X_csp = csp.transform(X)

    # Normalize the CSP features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_csp)

    # Separate normalized features by class
    X_nonfatigue = X_scaled[y == 0]
    X_fatigue = X_scaled[y == 1]

    # Plot the average normalized feature values across components
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_components + 1), np.mean(X_nonfatigue, axis=0), marker='o')
    plt.xlabel("CSP Component Number")
    plt.ylabel("Normalized Feature Value")
    plt.title("Alert State")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_components + 1), np.mean(X_fatigue, axis=0), marker='o')
    plt.xlabel("CSP Component Number")
    plt.ylabel("Normalized Feature Value")
    plt.title("Fatigue State")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
