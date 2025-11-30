from sklearn.metrics import confusion_matrix

def evaluate_model(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract TP, FP, FN, TN for class 1 (Fatigue)
    TN, FP, FN, TP = cm.ravel()
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision_fatigue = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_fatigue = TP / (TP + FN) if (TP + FN) > 0 else 0  # Same as sensitivity
    specificity_fatigue = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate
    f1_fatigue = 2 * (precision_fatigue * recall_fatigue) / (precision_fatigue + recall_fatigue) if (precision_fatigue + recall_fatigue) > 0 else 0

    # For Baseline (Alert) class (opposite of Fatigue)
    precision_baseline = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_baseline = TN / (TN + FP) if (TN + FP) > 0 else 0  # Sensitivity for baseline
    specificity_baseline = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Negative Rate for baseline
    f1_baseline = 2 * (precision_baseline * recall_baseline) / (precision_baseline + recall_baseline) if (precision_baseline + recall_baseline) > 0 else 0

    print(f"=== {name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fatigue Class (1): Precision={precision_fatigue:.4f}, Recall={recall_fatigue:.4f}, "
        f"Specificity={specificity_fatigue:.4f}, F1={f1_fatigue:.4f}")
    print(f"Baseline Class (0): Precision={precision_baseline:.4f}, Recall={recall_baseline:.4f}, "
        f"Specificity={specificity_baseline:.4f}, F1={f1_baseline:.4f}")

    metrics = {
        'accuracy': accuracy,
        'precision_fatigue': precision_fatigue,
        'recall_fatigue': recall_fatigue,
        'f1_fatigue': f1_fatigue,
        'precision_baseline': precision_baseline,
        'recall_baseline': recall_baseline,
        'f1_baseline': f1_baseline,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics