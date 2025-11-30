import numpy as np
from scipy.io import loadmat
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys 
import os
from Classes.config import Config
from src.evaluation import evaluate_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CSP_Analysis:

    def __init__(self, X, y, n_components=None, mode = 'all'):
        self.n_components = n_components
        self.csp = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
        self.X = X
        self.y = y
        self.mode = mode  # 'all' or 'selected'
    
    def transform_data(self, X):   
        return self.csp.transform(X)
    
    def preprocess_data(self, X, y):
        if X.ndim != 3:
            raise ValueError("X must have shape (n_trials, n_channels, n_samples)")

        # Convert X to float64 to avoid precision issues
        X = X.astype(np.float64)

        # Shuffle data to ensure random distribution
        np.random.seed(0)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        return X, y

    def split_data(self):

        #THREE-WAY SPLIT: Train (80%), Validation (10%), Test (10%)
        # First split into train+val and test
        self.X_trainval, self.X_test, self.y_trainval, self.y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=0, stratify=self.y
        )

        # Then split train+val into train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_trainval, self.y_trainval, test_size=1/9, random_state=0, stratify=self.y_trainval
        )  # 1/9 of 90% = 10% of original data

        print(f"The shape of the train set: {self.X_train.shape}")
        print(f"Training set: {self.X_train.shape[0]} samples ({self.X_train.shape[0]/len(self.X):.1%})")
        print(f"Validation set: {self.X_val.shape[0]} samples ({self.X_val.shape[0]/len(self.X):.1%})")
        print(f"Test set: {self.X_test.shape[0]} samples ({self.X_test.shape[0]/len(self.X):.1%})")
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    

    # ---- Model optimization with ALL CSP components ----
    # Step 1a: Find the best KNN parameter

    def finding_optimal_classifier(self, X_train_all_scaled, y_train, X_val_all_scaled, y_val):
        print("\n=== Finding Optimal KNN Parameters (All CSP Components) ===")
        self.best_k_all = 1
        self.best_accuracy_all = 0

        for k in range(1, 31, 2):  # Only odd values for k
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_all_scaled, y_train)
            val_accuracy = knn.score(X_val_all_scaled, y_val)
            
            print(f"k={k}, Validation accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > self.best_accuracy_all:
                self.best_accuracy_all = val_accuracy
                self.best_k_all = k

        print(f"\nSelected optimal k value (all CSP components): {self.best_k_all}")

        # Step 1b: Find the best SVM Linear parameters
        print("\n=== Finding Optimal Linear SVM Parameters (All CSP Components) ===")
        self.best_linear_params_all = {}
        best_linear_acc_all = 0

        C_values = Config.C_values
        for C in C_values:
            svm = SVC(kernel='linear', C=C)
            svm.fit(X_train_all_scaled, y_train)
            val_accuracy = svm.score(X_val_all_scaled, y_val)
            
            print(f"Linear SVM: C={C}, Validation accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_linear_acc_all:
                best_linear_acc_all = val_accuracy
                self.best_linear_params_all = {'C': C}

        print(f"\nSelected optimal Linear SVM parameters (all CSP components): {self.best_linear_params_all}")

        # Step 1c: Find the best SVM RBF parameters
        print("\n=== Finding Optimal RBF SVM Parameters (All CSP Components) ===")
        self.best_rbf_params_all = {}
        best_rbf_acc_all = 0

        C_values = Config.C_values
        gamma_values = Config.gamma_values

        for C in C_values:
            for gamma in gamma_values:
                svm = SVC(kernel='rbf', C=C, gamma=gamma)
                svm.fit(X_train_all_scaled, y_train)
                val_accuracy = svm.score(X_val_all_scaled, y_val)
                
                print(f"RBF SVM: C={C}, gamma={gamma}, Validation accuracy: {val_accuracy:.4f}")
                
                if val_accuracy > best_rbf_acc_all:
                    best_rbf_acc_all = val_accuracy
                    self.best_rbf_params_all = {'C': C, 'gamma': gamma}

        print(f"\nSelected optimal RBF SVM parameters (all CSP components): {self.best_rbf_params_all}")

        # Step 1d: Find the best Random Forest parameters
        print("\n=== Finding Optimal Random Forest Parameters (All CSP Components) ===")
        self.best_rf_params_all = {}
        best_rf_acc_all = 0

        for n_est in Config.n_estimators:
            for max_depth in Config.max_depth:
                rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=0)
                rf.fit(X_train_all_scaled, y_train)
                val_acc = rf.score(X_val_all_scaled, y_val)
                
                print(f"RF: n_estimators={n_est}, max_depth={max_depth}, Validation accuracy: {val_acc:.4f}")
                
                if val_acc > best_rf_acc_all:
                    best_rf_acc_all = val_acc
                    self.best_rf_params_all = {'n_estimators': n_est, 'max_depth': max_depth}
        print(f"\nSelected optimal RF parameters (all CSP components): {self.best_rf_params_all}")
        return self.best_k_all, self.best_linear_params_all, self.best_rbf_params_all, self.best_rf_params_all


    def run_csp_analysis(self):

        if self.mode == 'all':
            n_components = Config.num_components
            # ===== PART 1: USING ALL CSP COMPONENTS =====
            print("\n\n===== PART 1: USING ALL CSP COMPONENTS =====")

        else:
            print("The code for the selected components will come here !!")
            n_components = Config.n_component_list # the list of the selected components. 
            
            # Preprocess and split data first
            self.X, self.y = self.preprocess_data(self.X, self.y)
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data()

            print("\n=== Finding Optimal CSP Components ===")
            best_val_accuracy = 0
            self.best_component = n_components[0]
            
            for n_comp in n_components:
                # Fit CSP on training data
                csp = CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)
                csp.fit(self.X_train, self.y_train)
                
                # Transform train and validation data
                self.X_train_csp = csp.transform(self.X_train)
                self.X_val_csp = csp.transform(self.X_val)
                
                # Normalize features
                scaler = StandardScaler()
                self.X_train_scaled = scaler.fit_transform(self.X_train_csp)
                self.X_val_scaled = scaler.transform(self.X_val_csp)
                
                # Simple KNN to evaluate component performance
                knn = KNeighborsClassifier(n_neighbors=5)  # Using fixed k=5 just for component selection
                knn.fit(self.X_train_scaled, self.y_train)
                val_accuracy = knn.score(self.X_val_scaled, self.y_val)
                
                print(f"CSP components: {n_comp}, Validation accuracy: {val_accuracy:.4f}")
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.best_component = n_comp

            print(f"\nSelected optimal CSP components: {self.best_component}")
            n_components = self.best_component  # Use the selected component count

        self.X, self.y = self.preprocess_data(self.X, self.y)
        
        #splitting the data now
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data()
        
        # Apply CSP with all components (n_components=None or 30)
        self.csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        self.csp.fit(self.X_train, self.y_train)
        #Transforming the split data into CSP space
        self.X_train_csp_all = self.transform_data(self.X_train)
        self.X_val_csp_all = self.transform_data(self.X_val)

        # Normalize features
        self.evlaution_resultsscaler_all = StandardScaler()
        self.X_train_all_scaled = self.evlaution_resultsscaler_all.fit_transform(self.X_train_csp_all)
        self.X_val_all_scaled = self.evlaution_resultsscaler_all.transform(self.X_val_csp_all)

        best_k_all, best_linear_params_all, best_rbf_params_all, best_rf_params_all = self.finding_optimal_classifier(
            self.X_train_all_scaled, self.y_train, self.X_val_all_scaled, self.y_val
        )
        return best_k_all, best_linear_params_all, best_rbf_params_all, best_rf_params_all


    def run_evaluation(self):
        self.X_train_csp_all = self.transform_data(self.X_trainval) 
        self.X_test_csp_all = self.transform_data(self.X_test)
        self.X_trainval_all_scaled = self.evlaution_resultsscaler_all.fit_transform(self.X_train_csp_all)
        self.X_test_all_scaled = self.evlaution_resultsscaler_all.transform(self.X_test_csp_all)

        #KNN Metrics Evaluation !
        knn_all = KNeighborsClassifier(n_neighbors=self.best_k_all)
        knn_all.fit(self.X_trainval_all_scaled, self.y_trainval)
        y_test_pred_knn_all = knn_all.predict(self.X_test_all_scaled)
        knn_all_metrics = evaluate_model(self.y_test, y_test_pred_knn_all, "KNN (All CSP Components)")

        #SVM Linear Metrics Evaluation !
        svm_linear_all = SVC(kernel='linear', **self.best_linear_params_all)
        svm_linear_all.fit(self.X_trainval_all_scaled, self.y_trainval)
        y_test_pred_linear_all = svm_linear_all.predict(self.X_test_all_scaled)
        svm_linear_all_metrics = evaluate_model(self.y_test, y_test_pred_linear_all, "Linear SVM (All CSP Components)")

        # SVM RBF with all CSP components
        svm_rbf_all = SVC(kernel='rbf', **self.best_rbf_params_all)
        svm_rbf_all.fit(self.X_trainval_all_scaled, self.y_trainval)
        y_test_pred_rbf_all = svm_rbf_all.predict(self.X_test_all_scaled)
        svm_rbf_all_metrics = evaluate_model(self.y_test, y_test_pred_rbf_all, "RBF SVM (All CSP Components)")

        # Random Forest with all CSP components
        rf_all = RandomForestClassifier(**self.best_rf_params_all, random_state=0)
        rf_all.fit(self.X_trainval_all_scaled, self.y_trainval)
        y_test_pred_rf_all = rf_all.predict(self.X_test_all_scaled)
        rf_all_metrics = evaluate_model(self.y_test, y_test_pred_rf_all, "Random Forest (All CSP Components)")
        return knn_all_metrics, svm_linear_all_metrics, svm_rbf_all_metrics, rf_all_metrics
