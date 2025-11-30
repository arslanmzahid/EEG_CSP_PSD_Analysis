import sys
import os

# Add parent directory to path so we can import Classes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from Classes.config import Config
from csp_analysis import CSP_Analysis
import json
from data.creating_EEG_PSD_scripts import creating_EEG_PSD_scripts
import argparse
from eda.psd import  psd_eda
from eda.csp import csp_eda
from figures.figure_plotting import plot_accuracy_comparison


def load_data(file_path: str = Config.data_dir):
    """Load EEG data from a .mat file."""
    data = loadmat(file_path)
    X = data["EEGsample"]  # Shape (2022, 30, 384)
    X = X.astype(np.float64)
    y = data["substate"].ravel()  # Flatten labels
    return X, y


def main(arg_configuration: str = None):

    if arg_configuration == "Analysis_CSP":
        print("Running the Analysis CSP")
        csp_eda()
        exit()
    elif arg_configuration == "Analysis_PSD":
        print("Running the Analysis PSD")
        psd_eda()
        exit()
    # method = input("Enter method to run (CSP or PSD): ").strip().upper()
    # if method == "CSP":
    print("Starting the CSP Method Analysis...")
    # Load data
    X, y = load_data(Config.data_dir)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    mode = Config.mode  # 'all' or 'selected'
    #for all CSP analysis:
    csp_analysis = CSP_Analysis(X, y, n_components=Config.num_components, mode=mode)

    best_k_all, best_linear_params_all, best_rbf_params_all, best_rf_params_all = csp_analysis.run_csp_analysis()

    print(f"\n=== Optimal Classifier Parameters with {mode} CSP Components ===")
    print(f"Best KNN k ({mode} CSP): {best_k_all}")
    print(f"Best Linear SVM params ({mode} CSP): {best_linear_params_all}")
    print(f"Best RBF SVM params ({mode} CSP): {best_rbf_params_all}")
    print(f"Best Random Forest params ({mode} CSP): {best_rf_params_all}")
    #Evaluation and comparison can be added here as needed
    knn_all_metrics, svm_linear_all_metrics, svm_rbf_all_metrics, rf_all_metrics = csp_analysis.run_evaluation()
    dic = {
        'KNN': knn_all_metrics,
        'SVM_Linear': svm_linear_all_metrics,
        'SVM_RBF': svm_rbf_all_metrics,
        'Random_Forest': rf_all_metrics
    }
    Config.metrics_queue_csp.put(dic) #putting the metrics in the queue

    with open(f'csp_{mode}_metrics.json', 'w') as f:
        json.dump({
            'KNN': knn_all_metrics,
            'SVM_Linear': svm_linear_all_metrics,
            'SVM_RBF': svm_rbf_all_metrics,
            'Random_Forest': rf_all_metrics
        }, f, indent=4)
    # else:
    # Load PSD features
    print(f"First making features and saving to {Config.psd_dir}...")

    creating_EEG_PSD_scripts()
    print("Features created and saved.")


    print("Starting the PSD Methdod Analysis...")
    psd_data = loadmat(Config.psd_dir)
    X = psd_data["feature_matrix"]
    Y = psd_data["labels"]
    X = X.reshape(2022, -1)
    y = Y.ravel()

    #callng the PSD analysis class
    from psd_analysis import PSD_Analysis
    psd_analysis = PSD_Analysis(X, y, plot = True, mode=Config.mode)
    best_n_features, knn_metrics, svm_linear_metrics, svm_rbf_metrics, rf_metrics = psd_analysis.run_analysis()
    dic_psd = {
        'KNN': knn_metrics,
        'SVM_Linear': svm_linear_metrics,
        'SVM_RBF': svm_rbf_metrics,
        'Random_Forest': rf_metrics
    }
    Config.metrics_queue_psd.put(dic_psd) #putting the metrics in the queue

    print(f"\n=== PSD Analysis Results ===")
    print(f"Best number of features: {best_n_features}")
    
    # Dump PSD metrics to JSON
    with open(f'psd_{Config.mode}_metrics.json', 'w') as f:
        json.dump({
            'best_n_features': best_n_features,
            'KNN': knn_metrics,
            'SVM_Linear': svm_linear_metrics,
            'SVM_RBF': svm_rbf_metrics,
            'Random_Forest': rf_metrics
        }, f, indent=4) 


    plot_accuracy_comparison()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CSP or PSD analysis on EEG data.")
    parser.add_argument('--configuration', choices=["Analysis_CSP","Analysis_PSD"], type=str, help="Path to configuration file (not implemented yet).")
    args = parser.parse_args()


    main(args.configuration)