import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from Classes.config import Config


class PSD_Analysis:
    def __init__(self, X, y, plot, mode='all'):
        self.X = X
        self.y = y
        self.plot = plot
        self.mode = mode


    def shuffling_data(self, X, y):
        # Shuffle data to ensure random distribution
        np.random.seed(0)
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]
        return X, y

    def split_and_normalize(self, X, y):
        # THREE-WAY SPLIT: Train (80%), Validation (10%), Test (10%)
        # First split into train+val and test
        self.X_trainval, self.X_test, self.y_trainval, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=0, stratify=y
        )

        # Then split train+val into train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_trainval, self.y_trainval, test_size=1/9, random_state=0, stratify=self.y_trainval
        )  # 1/9 of 90% = 10% of original data


        print(f"Training set: {self.X_train.shape[0]} samples ({self.X_train.shape[0]/len(X):.1%})")
        print(f"Validation set: {self.X_val.shape[0]} samples ({self.X_val.shape[0]/len(X):.1%})")
        print(f"Test set: {self.X_test.shape[0]} samples ({self.X_test.shape[0]/len(X):.1%})")
        print(f"Total features: {self.X.shape[1]}")
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def feature_selection_and_model_optimization(self):
        # Normalize features using training data statistics
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_val_scaled = scaler.transform(self.X_val)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.X_trainval_scaled = scaler.transform(self.X_trainval)

        # ===== FEATURE SELECTION USING RANDOM FOREST =====
        print("\n=== Feature Selection Using Random Forest ===")

        # Train a random forest to get feature importances
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=0)
        rf_selector.fit(self.X_train_scaled, self.y_train)

        # Get feature importances and sort features by importance
        feature_importances = rf_selector.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]  # Sort in descending order

        if self.plot == True:
            # Plot feature importances
            plt.figure(figsize=(10, 6))
            plt.bar(range(self.X_train.shape[1]), feature_importances[sorted_idx])
            plt.xlabel('Feature Index (sorted by importance)')
            plt.ylabel('Feature Importance')
            plt.title('Random Forest Feature Importances')
            plt.xticks(range(0, self.X_train.shape[1], 10))
            plt.tight_layout()
            plt.savefig('feature_importances.png')
            plt.close()

            
        # Try different numbers of top features
        n_features_options = Config.n_features_options
        validation_scores = []

        print("\nEvaluating different feature subset sizes...")
        for n_features in n_features_options:
            # Select top n_features
            top_features = sorted_idx[:n_features]
            
            # Create datasets with only the selected features
            X_train_selected = self.X_train_scaled[:, top_features]
            X_val_selected = self.X_val_scaled[:, top_features]
            
            # Train a Random Forest with the selected features
            rf = RandomForestClassifier(n_estimators=100, random_state=0)
            rf.fit(X_train_selected, self.y_train)
            
            # Evaluate on validation set
            val_score = rf.score(X_val_selected, self.y_val)
            validation_scores.append(val_score)
            print(f"Top {n_features} features: Validation accuracy = {val_score:.4f}")

        # Find best number of features
        best_n_features_idx = np.argmax(validation_scores)
        best_n_features = n_features_options[best_n_features_idx]
        print(f"\nBest number of features: {best_n_features} with validation accuracy: {validation_scores[best_n_features_idx]:.4f}")

        # Get the indices of the best features
        best_features = sorted_idx[:best_n_features]
        if self.plot == True:
            # Plot validation accuracy vs number of features
            plt.figure(figsize=(10, 6))
            plt.plot(n_features_options, validation_scores, marker='o')
            plt.axvline(x=best_n_features, color='r', linestyle='--', label=f'Best: {best_n_features} features')
            plt.xlabel('Number of Top Features')
            plt.ylabel('Validation Accuracy')
            plt.title('Feature Selection: Validation Accuracy vs Number of Features')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig('feature_selection_curve.png')
            plt.close()

        # Create datasets with only the selected best features
        self.X_train_best = self.X_train_scaled[:, best_features]
        self.X_val_best = self.X_val_scaled[:, best_features]
        self.X_trainval_best = self.X_trainval_scaled[:, best_features]
        self.X_test_best = self.X_test_scaled[:, best_features]

        # Print feature indices that were selected
        print(f"Selected feature indices: {best_features[:10]}... (showing first 10)")
        return self.X_train_best, self.X_val_best, self.X_test_best, best_features

    def model_optimization_and_evaluation(self):
        # Initialize variables for both modes
        best_k_all = 1
        best_linear_params_all = {}
        best_rbf_params_all = {}
        best_rf_params_all = {}
        best_k = 1
        best_linear_params = {}
        best_rbf_params = {}
        best_rf_params = {}
        
        # ===== MODEL OPTIMIZATION WITH ALL FEATURES =====
        if self.mode == 'all':
            print("\n===== PART 1: MODEL OPTIMIZATION WITH ALL FEATURES =====")

            # Step 1: Find the best KNN parameter using training and validation data
            print("\n=== Finding Optimal KNN Parameters (All Features) ===")
            best_k_all = 1
            best_accuracy_all = 0
            accuracy_values_all = []

            for k in range(1, 31, 2):  # Only odd values for k
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(self.X_train_scaled, self.y_train)
                val_accuracy = knn.score(self.X_val_scaled, self.y_val)
                accuracy_values_all.append(val_accuracy)
    
            print(f"k={k}, Validation accuracy: {val_accuracy:.4f}")
        
            if val_accuracy > best_accuracy_all:
                best_accuracy_all = val_accuracy
                best_k_all = k
            print(f"\nSelected optimal k value (all features): {best_k_all}")

            # Step 2: Find the best SVM Linear parameters (all features)
            print("\n=== Finding Optimal Linear SVM Parameters (All Features) ===")
            best_linear_params_all = {}
            best_linear_acc_all = 0

            C_values = Config.C_values
            for C in C_values:
                svm = SVC(kernel='linear', C=C)
                svm.fit(self.X_train_scaled, self.y_train)
                val_accuracy = svm.score(self.X_val_scaled, self.y_val)
                
                print(f"Linear SVM: C={C}, Validation accuracy: {val_accuracy:.4f}")
                
                if val_accuracy > best_linear_acc_all:
                    best_linear_acc_all = val_accuracy
                    best_linear_params_all = {'C': C}

            print(f"\nSelected optimal Linear SVM parameters (all features): {best_linear_params_all}")

            # Step 3: Find the best SVM RBF parameters (all features)
            print("\n=== Finding Optimal RBF SVM Parameters (All Features) ===")
            best_rbf_params_all = {}
            best_rbf_acc_all = 0

            C_values = Config.C_values
            gamma_values = Config.gamma_values

            for C in C_values:
                for gamma in gamma_values:
                    svm = SVC(kernel='rbf', C=C, gamma=gamma)
                    svm.fit(self.X_train_scaled, self.y_train)
                    val_accuracy = svm.score(self.X_val_scaled, self.y_val)
                    
                    print(f"RBF SVM: C={C}, gamma={gamma}, Validation accuracy: {val_accuracy:.4f}")
                    
                    if val_accuracy > best_rbf_acc_all:
                        best_rbf_acc_all = val_accuracy
                        best_rbf_params_all = {'C': C, 'gamma': gamma}

            print(f"\nSelected optimal RBF SVM parameters (all features): {best_rbf_params_all}")

            # Step 4: Find the best Random Forest parameters (all features)
            print("\n=== Finding Optimal Random Forest Parameters (All Features) ===")
            best_rf_params_all = {}
            best_rf_acc_all = 0

            for n_est in Config.n_estimators:
                for max_depth in Config.max_depth:
                    rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=0)
                    rf.fit(self.X_train_scaled, self.y_train)
                    val_acc = rf.score(self.X_val_scaled, self.y_val)
                    
                    print(f"RF: n_estimators={n_est}, max_depth={max_depth}, Validation accuracy: {val_acc:.4f}")
                    
                    if val_acc > best_rf_acc_all:
                        best_rf_acc_all = val_accuracy
                        best_rf_params_all = {'n_estimators': n_est, 'max_depth': max_depth}

            print(f"\nSelected optimal RF parameters (all features): {best_rf_params_all}")

        else:

            # ===== MODEL OPTIMIZATION WITH SELECTED FEATURES =====
            print("\n===== PART 2: MODEL OPTIMIZATION WITH SELECTED FEATURES =====")

            # Step 1: Find the best KNN parameter using training and validation data
            print("\n=== Finding Optimal KNN Parameters (Selected Features) ===")
            best_k = 1
            best_accuracy = 0
            accuracy_values = []

            for k in range(1, 31, 2):  # Only odd values for k
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(self.X_train_best, self.y_train)
                val_accuracy = knn.score(self.X_val_best, self.y_val)
                accuracy_values.append(val_accuracy)
                
                print(f"k={k}, Validation accuracy: {val_accuracy:.4f}")
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_k = k

            print(f"\nSelected optimal k value (selected features): {best_k}")

            # Step 2: Find the best SVM Linear parameters (selected features)
            print("\n=== Finding Optimal Linear SVM Parameters (Selected Features) ===")
            best_linear_params = {}
            best_linear_acc = 0

            C_values = Config.C_values
            for C in C_values:
                svm = SVC(kernel='linear', C=C)
                svm.fit(self.X_train_best, self.y_train)
                val_accuracy = svm.score(self.X_val_best, self.y_val)
                
                print(f"Linear SVM: C={C}, Validation accuracy: {val_accuracy:.4f}")
                
                if val_accuracy > best_linear_acc:
                    best_linear_acc = val_accuracy
                    best_linear_params = {'C': C}

            print(f"\nSelected optimal Linear SVM parameters (selected features): {best_linear_params}")

            # Step 3: Find the best SVM RBF parameters (selected features)
            print("\n=== Finding Optimal RBF SVM Parameters (Selected Features) ===")
            best_rbf_params = {}
            best_rbf_acc = 0

            C_values = Config.C_values
            gamma_values = Config.gamma_values

            for C in C_values:
                for gamma in gamma_values:
                    svm = SVC(kernel='rbf', C=C, gamma=gamma)
                    svm.fit(self.X_train_best, self.y_train)
                    val_accuracy = svm.score(self.X_val_best, self.y_val)
                    
                    print(f"RBF SVM: C={C}, gamma={gamma}, Validation accuracy: {val_accuracy:.4f}")
                    
                    if val_accuracy > best_rbf_acc:
                        best_rbf_acc = val_accuracy
                        best_rbf_params = {'C': C, 'gamma': gamma}

            print(f"\nSelected optimal RBF SVM parameters (selected features): {best_rbf_params}")

            # Step 4: Find the best Random Forest parameters (selected features)
            print("\n=== Finding Optimal Random Forest Parameters (Selected Features) ===")
            best_rf_params = {}
            best_rf_acc = 0

            for n_est in Config.n_estimators:
                for max_depth in Config.max_depth:
                    rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=0)
                    rf.fit(self.X_train_best, self.y_train)
                    val_acc = rf.score(self.X_val_best, self.y_val)
                    
                    print(f"RF: n_estimators={n_est}, max_depth={max_depth}, Validation accuracy: {val_acc:.4f}")
                    
                    if val_acc > best_rf_acc:
                        best_rf_acc = val_acc
                        best_rf_params = {'n_estimators': n_est, 'max_depth': max_depth}

            print(f"\nSelected optimal RF parameters (selected features): {best_rf_params}")
        return best_k_all, best_linear_params_all, best_rbf_params_all, best_rf_params_all, best_k, best_linear_params, best_rbf_params, best_rf_params


    def run_analysis(self):

        #shuffling the data 
        self.X, self.y = self.shuffling_data(self.X, self.y)
        # Split and normalize data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_and_normalize(self.X, self.y)

        # Feature selection and model optimization
        self.X_train_best, self.X_val_best, self.X_test_best, best_features = self.feature_selection_and_model_optimization()

        #model optimization and evaluation
        best_k_all, best_linear_params_all, best_rbf_params_all, best_rf_params_all, best_k, best_linear_params, best_rbf_params, best_rf_params = self.model_optimization_and_evaluation()
        
        # Store best parameters for evaluation
        self.best_k_all = best_k_all
        self.best_linear_params_all = best_linear_params_all
        self.best_rbf_params_all = best_rbf_params_all
        self.best_rf_params_all = best_rf_params_all
        self.best_n_features = 50  # From feature selection
        
        # Run evaluation on test set
        knn_all_metrics, svm_linear_all_metrics, svm_rbf_all_metrics, rf_all_metrics = self.run_evaluation()
        
        return self.best_n_features, knn_all_metrics, svm_linear_all_metrics, svm_rbf_all_metrics, rf_all_metrics

    def run_evaluation(self):
        """Evaluate all models on the test set using best parameters."""
        from evaluation import evaluate_model
        
        # Prepare test data with best features
        self.X_trainval_best = np.vstack([self.X_train_best, self.X_val_best])
        self.y_trainval_best = np.vstack([self.y_train.reshape(-1, 1), self.y_val.reshape(-1, 1)]).ravel()
        
        # KNN Evaluation
        knn_all = KNeighborsClassifier(n_neighbors=self.best_k_all)
        knn_all.fit(self.X_trainval_best, self.y_trainval_best)
        y_test_pred_knn = knn_all.predict(self.X_test_best)
        knn_all_metrics = evaluate_model(self.y_test, y_test_pred_knn, "KNN (PSD Selected Features)")
        
        # Linear SVM Evaluation
        svm_linear_all = SVC(kernel='linear', **self.best_linear_params_all)
        svm_linear_all.fit(self.X_trainval_best, self.y_trainval_best)
        y_test_pred_linear = svm_linear_all.predict(self.X_test_best)
        svm_linear_all_metrics = evaluate_model(self.y_test, y_test_pred_linear, "Linear SVM (PSD Selected Features)")
        
        # RBF SVM Evaluation
        svm_rbf_all = SVC(kernel='rbf', **self.best_rbf_params_all)
        svm_rbf_all.fit(self.X_trainval_best, self.y_trainval_best)
        y_test_pred_rbf = svm_rbf_all.predict(self.X_test_best)
        svm_rbf_all_metrics = evaluate_model(self.y_test, y_test_pred_rbf, "RBF SVM (PSD Selected Features)")
        
        # Random Forest Evaluation
        rf_all = RandomForestClassifier(**self.best_rf_params_all, random_state=0)
        rf_all.fit(self.X_trainval_best, self.y_trainval_best)
        y_test_pred_rf = rf_all.predict(self.X_test_best)
        rf_all_metrics = evaluate_model(self.y_test, y_test_pred_rf, "Random Forest (PSD Selected Features)")
        
        return knn_all_metrics, svm_linear_all_metrics, svm_rbf_all_metrics, rf_all_metrics


    # Helper function to evaluate and return metrics
    def evaluate_model(self, y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        f1_per_class = f1_score(y_true, y_pred, average=None)  # Gives F1 for each class
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        print(f"{name} - Accuracy: {acc:.4f}, F1 (Baseline): {f1_per_class[0]:.4f}, "
            f"F1 (Fatigue): {f1_per_class[1]:.4f}, Macro F1: {macro_f1:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)

        # === Models with ALL features ===
        print("\n=== Final Test Results with ALL Features ===")

        # KNN with best parameter (all features)
        knn_all = KNeighborsClassifier(n_neighbors=self.best_k_all)
        knn_all.fit(self.X_trainval_scaled, self.Y_trainval)
        Y_test_pred_knn_all = knn_all.predict(self.X_test_scaled)
        knn_all_metrics = self.evaluate_model(self.Y_test, Y_test_pred_knn_all, "KNN (All Features)")

        # SVM Linear with optimized parameters (all features)
        svm_linear_all = SVC(kernel='linear', **self.best_linear_params_all)
        svm_linear_all.fit(self.X_trainval_scaled, self.Y_trainval)
        Y_test_pred_linear_all = svm_linear_all.predict(self.X_test_scaled)
        svm_linear_all_metrics = self.evaluate_model(self.Y_test, Y_test_pred_linear_all, "Linear SVM (All Features)")

        # SVM RBF with optimized parameters (all features)
        svm_rbf_all = SVC(kernel='rbf', **self.best_rbf_params_all)
        svm_rbf_all.fit(self.X_trainval_scaled, self.Y_trainval)
        Y_test_pred_rbf_all = svm_rbf_all.predict(self.X_test_scaled)
        svm_rbf_all_metrics = self.evaluate_model(self.Y_test, Y_test_pred_rbf_all, "RBF SVM (All Features)")

        # Final Random Forest (all features)
        rf_all = RandomForestClassifier(**self.best_rf_params_all, random_state=0)
        rf_all.fit(self.X_trainval_scaled, self.Y_trainval)
        Y_test_pred_rf_all = rf_all.predict(self.X_test_scaled)
        rf_all_metrics = self.evaluate_model(self.Y_test, Y_test_pred_rf_all, "Random Forest (All Features)")

        # === Models with SELECTED features ===
        print(f"\n=== Final Test Results with {Config.mode} Features ===")

        # KNN with best parameter (selected features)
        knn_sel = KNeighborsClassifier(n_neighbors=self.best_k)
        knn_sel.fit(self.X_trainval_best, self.Y_trainval)
        Y_test_pred_knn_sel = knn_sel.predict(self.X_test_best)
        knn_sel_metrics = self.evaluate_model(self.Y_test, Y_test_pred_knn_sel, f"KNN ({Config.mode} Features)")

        # SVM Linear with optimized parameters (selected features)
        svm_linear_sel = SVC(kernel='linear', **self.best_linear_params)
        svm_linear_sel.fit(self.X_trainval_best, self.Y_trainval)
        Y_test_pred_linear_sel = svm_linear_sel.predict(self.X_test_best)
        svm_linear_sel_metrics = self.evaluate_model(self.Y_test, Y_test_pred_linear_sel, f"Linear SVM ({Config.mode} Features)")

        # SVM RBF with optimized parameters (selected features)
        svm_rbf_sel = SVC(kernel='rbf', **self.best_rbf_params)
        svm_rbf_sel.fit(self.X_trainval_best, self.Y_trainval)
        Y_test_pred_rbf_sel = svm_rbf_sel.predict(self.X_test_best)
        svm_rbf_sel_metrics = self.evaluate_model(self.Y_test, Y_test_pred_rbf_sel, f"RBF SVM ({Config.mode} Features)")

        # Final Random Forest (selected features)
        rf_sel = RandomForestClassifier(**self.best_rf_params, random_state=0)
        rf_sel.fit(self.X_trainval_best, self.Y_trainval)
        Y_test_pred_rf_sel = rf_sel.predict(self.X_test_best)
        rf_sel_metrics = self.evaluate_model(self.Y_test, Y_test_pred_rf_sel, f"Random Forest ({Config.mode} Features)")

        return {
            'accuracy': acc, 
            'f1_baseline': f1_per_class[0], 
            'f1_fatigue': f1_per_class[1],
            'macro_f1': macro_f1,
            'confusion_matrix': cm
        }


# # === COMPARISON TABLE ===
# print("\n=== Feature Selection Impact: Performance Comparison ===")
# models = ["KNN", "Linear SVM", "RBF SVM", "Random Forest"]
# metrics = ["Accuracy", "F1 (Baseline)", "F1 (Fatigue)", "Macro F1"]

# print("\nModel           | All Features         | Selected Features   | Difference (Sel-All)")
# print("----------------------------------------------------------------------------")

# # KNN comparison
# print(f"KNN - Accuracy  | {knn_all_metrics['accuracy']:.4f}                | {knn_sel_metrics['accuracy']:.4f}                | {knn_sel_metrics['accuracy'] - knn_all_metrics['accuracy']:.4f}")
# print(f"KNN - F1 Base   | {knn_all_metrics['f1_baseline']:.4f}                | {knn_sel_metrics['f1_baseline']:.4f}                | {knn_sel_metrics['f1_baseline'] - knn_all_metrics['f1_baseline']:.4f}")
# print(f"KNN - F1 Fatigue| {knn_all_metrics['f1_fatigue']:.4f}                | {knn_sel_metrics['f1_fatigue']:.4f}                | {knn_sel_metrics['f1_fatigue'] - knn_all_metrics['f1_fatigue']:.4f}")

# # Linear SVM comparison
# print(f"Lin SVM - Acc   | {svm_linear_all_metrics['accuracy']:.4f}                | {svm_linear_sel_metrics['accuracy']:.4f}                | {svm_linear_sel_metrics['accuracy'] - svm_linear_all_metrics['accuracy']:.4f}")
# print(f"Lin SVM - F1 B  | {svm_linear_all_metrics['f1_baseline']:.4f}                | {svm_linear_sel_metrics['f1_baseline']:.4f}                | {svm_linear_sel_metrics['f1_baseline'] - svm_linear_all_metrics['f1_baseline']:.4f}")
# print(f"Lin SVM - F1 F  | {svm_linear_all_metrics['f1_fatigue']:.4f}                | {svm_linear_sel_metrics['f1_fatigue']:.4f}                | {svm_linear_sel_metrics['f1_fatigue'] - svm_linear_all_metrics['f1_fatigue']:.4f}")

# # RBF SVM comparison
# print(f"RBF SVM - Acc   | {svm_rbf_all_metrics['accuracy']:.4f}                | {svm_rbf_sel_metrics['accuracy']:.4f}                | {svm_rbf_sel_metrics['accuracy'] - svm_rbf_all_metrics['accuracy']:.4f}")
# print(f"RBF SVM - F1 B  | {svm_rbf_all_metrics['f1_baseline']:.4f}                | {svm_rbf_sel_metrics['f1_baseline']:.4f}                | {svm_rbf_sel_metrics['f1_baseline'] - svm_rbf_all_metrics['f1_baseline']:.4f}")
# print(f"RBF SVM - F1 F  | {svm_rbf_all_metrics['f1_fatigue']:.4f}                | {svm_rbf_sel_metrics['f1_fatigue']:.4f}                | {svm_rbf_sel_metrics['f1_fatigue'] - svm_rbf_all_metrics['f1_fatigue']:.4f}")

# # RF comparison
# print(f"RF - Accuracy   | {rf_all_metrics['accuracy']:.4f}                | {rf_sel_metrics['accuracy']:.4f}                | {rf_sel_metrics['accuracy'] - rf_all_metrics['accuracy']:.4f}")
# print(f"RF - F1 Base    | {rf_all_metrics['f1_baseline']:.4f}                | {rf_sel_metrics['f1_baseline']:.4f}                | {rf_sel_metrics['f1_baseline'] - rf_all_metrics['f1_baseline']:.4f}")
# print(f"RF - F1 Fatigue | {rf_all_metrics['f1_fatigue']:.4f}                | {rf_sel_metrics['f1_fatigue']:.4f}                | {rf_sel_metrics['f1_fatigue'] - rf_all_metrics['f1_fatigue']:.4f}")

# === VISUALIZATION ===
# Visualize confusion matrices for all models
'''
plt.figure(figsize=(16, 12))
all_models = [
    ('KNN (All)', knn_all_metrics['confusion_matrix']), 
    ('KNN (Selected)', knn_sel_metrics['confusion_matrix']),
    ('Linear SVM (All)', svm_linear_all_metrics['confusion_matrix']), 
    ('Linear SVM (Selected)', svm_linear_sel_metrics['confusion_matrix']),
    ('RBF SVM (All)', svm_rbf_all_metrics['confusion_matrix']), 
    ('RBF SVM (Selected)', svm_rbf_sel_metrics['confusion_matrix']),
    ('Random Forest (All)', rf_all_metrics['confusion_matrix']), 
    ('Random Forest (Selected)', rf_sel_metrics['confusion_matrix'])
]

for i, (name, cm) in enumerate(all_models, 1):
    plt.subplot(4, 2, i)
    
    # Display the confusion matrix
    plt.imshow(cm, cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.colorbar()
    
    # Remove the axis ticks
    plt.xticks([])
    plt.yticks([])
    
    # Annotate confusion matrix with values
    thresh = cm.max() / 2
    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            plt.text(k, j, format(cm[j, k], 'd'), ha="center", va="center", 
                     color="white" if cm[j, k] > thresh else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png')
plt.close()

# Compare metrics across models visually
metrics_names = ['Accuracy', 'F1 (Baseline)', 'F1 (Fatigue)']
model_names = ['KNN', 'Linear SVM', 'RBF SVM', 'Random Forest']

metrics_all = np.array([
    [knn_all_metrics['accuracy'], knn_all_metrics['f1_baseline'], knn_all_metrics['f1_fatigue']],
    [svm_linear_all_metrics['accuracy'], svm_linear_all_metrics['f1_baseline'], svm_linear_all_metrics['f1_fatigue']],
    [svm_rbf_all_metrics['accuracy'], svm_rbf_all_metrics['f1_baseline'], svm_rbf_all_metrics['f1_fatigue']],
    [rf_all_metrics['accuracy'], rf_all_metrics['f1_baseline'], rf_all_metrics['f1_fatigue']]
])

metrics_sel = np.array([
    [knn_sel_metrics['accuracy'], knn_sel_metrics['f1_baseline'], knn_sel_metrics['f1_fatigue']],
    [svm_linear_sel_metrics['accuracy'], svm_linear_sel_metrics['f1_baseline'], svm_linear_sel_metrics['f1_fatigue']],
    [svm_rbf_sel_metrics['accuracy'], svm_rbf_sel_metrics['f1_baseline'], svm_rbf_sel_metrics['f1_fatigue']],
    [rf_sel_metrics['accuracy'], rf_sel_metrics['f1_baseline'], rf_sel_metrics['f1_fatigue']]
])

# Create grouped bar plots for each metric
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
x = np.arange(len(model_names))
width = 0.35

for i, metric_name in enumerate(metrics_names):
    ax = axes[i]
    ax.bar(x - width/2, metrics_all[:, i], width, label='All Features')
    ax.bar(x + width/2, metrics_sel[:, i], width, label='Selected Features')
    
    ax.set_xlabel('Models')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('metrics_comparison.png')
plt.close()

# Plot relative importance of the selected features
if best_n_features < 120:  # Only show if we actually did feature selection
    plt.figure(figsize=(10, 6))
    selected_importances = feature_importances[best_features]
    sorted_selected_idx = np.argsort(selected_importances)[::-1]
    
    plt.bar(range(len(best_features)), selected_importances[sorted_selected_idx])
    plt.xlabel('Selected Feature Index (sorted by importance)')
    plt.ylabel('Feature Importance')
    plt.title(f'Feature Importances of Selected Top {best_n_features} Features')
    plt.tight_layout()
    plt.savefig('selected_features_importance.png')
    plt.close()

# Print summary of findings
print("\n=== Summary of Findings ===")
print(f"Total number of features in dataset: {X.shape[1]}")
print(f"Optimal number of features selected: {best_n_features} ({best_n_features/X.shape[1]:.1%} of total)")

# Calculate average performance impact across models
accuracy_diffs = [
    knn_sel_metrics['accuracy'] - knn_all_metrics['accuracy'],
    svm_linear_sel_metrics['accuracy'] - svm_linear_all_metrics['accuracy'],
    svm_rbf_sel_metrics['accuracy'] - svm_rbf_all_metrics['accuracy'],
    rf_sel_metrics['accuracy'] - rf_all_metrics['accuracy']
]

avg_accuracy_impact = np.mean(accuracy_diffs)
print(f"Average impact on accuracy: {avg_accuracy_impact:.4f}")

if avg_accuracy_impact > 0:
    print("Feature selection IMPROVED model performance on average")
elif avg_accuracy_impact < 0:
    print("Feature selection REDUCED model performance on average")
else:
    print("Feature selection had NO IMPACT on model performance on average")

print("\n=== Selected Hyperparameters ===")
print(f"Number of selected features: {best_n_features} out of {X.shape[1]}")
print("All Features Models:")
print(f"  Best KNN n_neighbors: {best_k_all}")
print(f"  Best Linear SVM parameters: {best_linear_params_all}")
print(f"  Best RBF SVM parameters: {best_rbf_params_all}")
print(f"  Best Random Forest parameters: {best_rf_params_all}")

print("\nSelected Features Models:")
print(f"  Best KNN n_neighbors: {best_k}")
print(f"  Best Linear SVM parameters: {best_linear_params}")
print(f"  Best RBF SVM parameters: {best_rbf_params}")
print(f"  Best Random Forest parameters: {best_rf_params}")

# Identify best overall model
all_metrics = {
    'KNN (All)': knn_all_metrics,
    'KNN (Selected)': knn_sel_metrics,
    'Linear SVM (All)': svm_linear_all_metrics,
    'Linear SVM (Selected)': svm_linear_sel_metrics,
    'RBF SVM (All)': svm_rbf_all_metrics,
    'RBF SVM (Selected)': svm_rbf_sel_metrics,
    'Random Forest (All)': rf_all_metrics,
    'Random Forest (Selected)': rf_sel_metrics
}

# Find best model by accuracy
best_model = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest overall model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")

# Computational cost comparison if desired
print("\n=== Computational Advantage of Feature Selection ===")
print(f"Feature reduction: {X.shape[1]} → {best_n_features} features ({best_n_features/X.shape[1]:.1%} of original)")
print("This reduction could lead to significant training and inference speed improvements")
print("especially for large datasets or when deploying models with limited computational resources.")
'''