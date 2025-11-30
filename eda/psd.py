import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os



def psd_eda():
    # Load your data and labels
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "EEG_PSD_Features.mat")
    data = loadmat(data_path)
    X = data["feature_matrix"]
    Y = data["labels"]
    X = X.reshape(2022, -1)

    # Shuffle data to ensure random distribution
    np.random.seed(0)
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    Y = Y[indices]

    # THREE-WAY SPLIT: Train (80%), Validation (10%), Test (10%)
    # First split into train+val and test
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=0, stratify=Y
    )

    # Then split train+val into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=1/9, random_state=0, stratify=Y_trainval
    )  # 1/9 of 90% = 10% of original data

    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    print(f"Total features: {X.shape[1]}")

    # Normalize features using training data statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ===== FEATURE SELECTION USING RANDOM FOREST =====
    print("\n=== Feature Selection Using Random Forest ===")

    # Train a random forest to get feature importances
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=0)
    rf_selector.fit(X_train_scaled, Y_train)

    # Get feature importances and sort features by importance
    feature_importances = rf_selector.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]  # Sort in descending order

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), feature_importances[sorted_idx])
    plt.xlabel('Feature Index (sorted by importance)')
    plt.ylabel('Feature Importance')
    plt.title('Random Forest Feature Importances')
    plt.xticks(range(0, X_train.shape[1], 10))
    plt.tight_layout()
    plt.show()

    # Try different numbers of top features
    n_features_options = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    validation_scores = []

    print("\nEvaluating different feature subset sizes...")
    for n_features in n_features_options:
        # Select top n_features
        top_features = sorted_idx[:n_features]
        
        # Create datasets with only the selected features
        X_train_selected = X_train_scaled[:, top_features]
        X_val_selected = X_val_scaled[:, top_features]
        
        # Train a Random Forest with the selected features
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(X_train_selected, Y_train)
        
        # Evaluate on validation set
        val_score = rf.score(X_val_selected, Y_val)
        validation_scores.append(val_score)
        print(f"Top {n_features} features: Validation accuracy = {val_score:.4f}")

    # Find best number of features
    best_n_features_idx = np.argmax(validation_scores)
    best_n_features = n_features_options[best_n_features_idx]
    print(f"\nBest number of features: {best_n_features} with validation accuracy: {validation_scores[best_n_features_idx]:.4f}")

    # Get the indices of the best features
    best_features = sorted_idx[:best_n_features]

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
    plt.show()

    # Use only the selected features for subsequent analysis
    print(f"\nUsing top {best_n_features} features for all models...")

    # Create datasets with only the selected best features
    X_train_best = X_train_scaled[:, best_features]
    X_val_best = X_val_scaled[:, best_features]
    X_trainval_scaled = scaler.transform(X_trainval)
    X_trainval_best = X_trainval_scaled[:, best_features]
    X_test_best = X_test_scaled[:, best_features]

    # Print feature indices that were selected
    print(f"Selected feature indices: {best_features[:10]}... (showing first 10)")

    # Continue with the model training pipeline but using the selected features
    # Step 1: Find the best KNN parameter using training and validation data
    print("\n=== Finding Optimal KNN Parameters ===")
    best_k = 1
    best_accuracy = 0
    accuracy_values = []

    for k in range(1, 31, 2):  # Only odd values for k
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_best, Y_train)
        val_accuracy = knn.score(X_val_best, Y_val)
        accuracy_values.append(val_accuracy)
        
        print(f"k={k}, Validation accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_k = k

    print(f"\nSelected optimal k value: {best_k}")

    # Step 2: Find the best SVM Linear parameters
    print("\n=== Finding Optimal Linear SVM Parameters ===")
    best_linear_params = {}
    best_linear_acc = 0

    C_values = [0.1, 1, 10, 100]
    for C in C_values:
        svm = SVC(kernel='linear', C=C)
        svm.fit(X_train_best, Y_train)
        val_accuracy = svm.score(X_val_best, Y_val)
        
        print(f"Linear SVM: C={C}, Validation accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_linear_acc:
            best_linear_acc = val_accuracy
            best_linear_params = {'C': C}

    print(f"\nSelected optimal Linear SVM parameters: {best_linear_params}")

    # Step 3: Find the best SVM RBF parameters
    print("\n=== Finding Optimal RBF SVM Parameters ===")
    best_rbf_params = {}
    best_rbf_acc = 0

    C_values = [0.1, 1, 10, 100]
    gamma_values = ['scale', 'auto', 0.01, 0.1, 1]

    for C in C_values:
        for gamma in gamma_values:
            svm = SVC(kernel='rbf', C=C, gamma=gamma)
            svm.fit(X_train_best, Y_train)
            val_accuracy = svm.score(X_val_best, Y_val)
            
            print(f"RBF SVM: C={C}, gamma={gamma}, Validation accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_rbf_acc:
                best_rbf_acc = val_accuracy
                best_rbf_params = {'C': C, 'gamma': gamma}

    print(f"\nSelected optimal RBF SVM parameters: {best_rbf_params}")

    # Step 4: Find the best Random Forest parameters
    print("\n=== Finding Optimal Random Forest Parameters ===")
    best_rf_params = {}
    best_rf_acc = 0

    for n_est in [50, 100, 200]:
        for max_depth in [None, 10, 20, 30]:
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=0)
            rf.fit(X_train_best, Y_train)
            val_acc = rf.score(X_val_best, Y_val)
            
            print(f"RF: n_estimators={n_est}, max_depth={max_depth}, Validation accuracy: {val_acc:.4f}")
            
            if val_acc > best_rf_acc:
                best_rf_acc = val_acc
                best_rf_params = {'n_estimators': n_est, 'max_depth': max_depth}

    print(f"\nSelected optimal RF parameters: {best_rf_params}")

    # Step 5: Now that we've selected parameters using validation data,
    # retrain on combined train+validation and evaluate on test
    print("\n=== Training Final Models ===")

    # KNN with best parameter
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_trainval_best, Y_trainval)
    Y_test_pred_knn = knn_final.predict(X_test_best)

    # SVM Linear with optimized parameters
    svm_linear = SVC(kernel='linear', **best_linear_params)
    svm_linear.fit(X_trainval_best, Y_trainval)
    Y_test_pred_linear = svm_linear.predict(X_test_best)

    # SVM RBF with optimized parameters
    svm_rbf = SVC(kernel='rbf', **best_rbf_params)
    svm_rbf.fit(X_trainval_best, Y_trainval)
    Y_test_pred_rbf = svm_rbf.predict(X_test_best)

    # Final Random Forest
    rf_final = RandomForestClassifier(**best_rf_params, random_state=0)
    rf_final.fit(X_trainval_best, Y_trainval)
    Y_test_pred_rf = rf_final.predict(X_test_best)

    # Evaluate all models
    def evaluate_model(y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        f1_per_class = f1_score(y_true, y_pred, average=None)  # Gives F1 for each class
        print(f"{name} - Accuracy: {acc:.4f}, F1 (Baseline): {f1_per_class[0]:.4f}, F1 (Fatigue): {f1_per_class[1]:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        return cm

    # Final test results (only evaluated once)
    print("\n=== Final Test Results with Feature Selection ===")
    cm_knn = evaluate_model(Y_test, Y_test_pred_knn, "KNN")
    cm_svm_lin = evaluate_model(Y_test, Y_test_pred_linear, "Linear SVM")
    cm_svm_rbf = evaluate_model(Y_test, Y_test_pred_rbf, "RBF SVM")
    cm_rf = evaluate_model(Y_test, Y_test_pred_rf, "Random Forest")

    # Plot confusion matrices
    plt.figure(figsize=(12, 10))
    models = [('KNN', cm_knn, Y_test_pred_knn), ('Linear SVM', cm_svm_lin, Y_test_pred_linear),
            ('RBF SVM', cm_svm_rbf, Y_test_pred_rbf), ('Random Forest', cm_rf, Y_test_pred_rf)]

    for i, (name, cm, y_pred) in enumerate(models, 1):
        plt.subplot(2, 2, i)
        accuracy = accuracy_score(Y_test, y_pred)
        
        # Display the confusion matrix
        plt.imshow(cm, cmap='Blues')
        plt.title(f'{name} Confusion Matrix\nAccuracy: {accuracy:.4f}')
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
    plt.show()

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
        plt.show()

    # Print hyperparameters for reference
    print("\n=== Selected Hyperparameters with Feature Selection ===")
    print(f"Number of selected features: {best_n_features} out of 120")
    print(f"Best KNN n_neighbors: {best_k}")
    print(f"Best Linear SVM parameters: {best_linear_params}")
    print(f"Best RBF SVM parameters: {best_rbf_params}")
    print(f"Best Random Forest parameters: {best_rf_params}")