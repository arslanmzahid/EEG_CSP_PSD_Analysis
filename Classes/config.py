import os
from queue import Queue

class Config:
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_dir, "data")
    data_dir = os.path.join(data_path, "dataset.mat")

    psd_dir = os.path.join(data_path, "EEG_PSD_Features.mat")

    num_components = 30  # Number of CSP components to use
    n_component_list = [4, 6, 8, 10, 12]
    C_values = [0.1, 1, 10, 100]
    gamma_values = ['scale', 'auto', 0.01, 0.1, 1]
    n_estimators = [50, 100, 200]
    max_depth = [None, 10, 20]
    n_features_options = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    mode = 'selected'  # 'all' or 'selected'

    metrics_queue_csp = Queue()  # Queue to store metrics during evaluation
    metrics_queue_psd = Queue()  # Queue to store metrics during evaluation