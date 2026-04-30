import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.model import LightweightECGNet
from src.utils import ECGDataset

def evaluate_model(X_test, y_test, df_test, model_path='results/best_model.pth', device='cpu'):
    print("Evaluating model on test set...")
    
    # Load model
    model = LightweightECGNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_dataset = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Handle single batch case
            if probs.ndim == 0:
                probs = np.array([probs])
            
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    # Compute metrics safely for small sample sizes
    metrics = {}
    
    try:
        metrics['AUROC'] = roc_auc_score(y_test, all_probs)
    except ValueError:
        metrics['AUROC'] = 0.5
        
    try:
        metrics['AUPRC'] = average_precision_score(y_test, all_probs)
    except ValueError:
        metrics['AUPRC'] = 0.5
        
    metrics['Accuracy'] = accuracy_score(y_test, all_preds)
    metrics['Precision'] = precision_score(y_test, all_preds, zero_division=0)
    metrics['Recall'] = recall_score(y_test, all_preds, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, all_preds).ravel()
        metrics['Specificity'] = tn / (tn + fp + 1e-8)
    except ValueError:
        metrics['Specificity'] = 0.0
        
    metrics['F1_Score'] = f1_score(y_test, all_preds, zero_division=0)
    
    # Save metrics
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to results/metrics.json")
    
    # Save test predictions
    df_test['predicted_probability'] = all_probs
    df_test['predicted_label'] = all_preds
    df_test['true_label'] = y_test
    df_test[['true_label', 'predicted_probability', 'predicted_label', 'strat_fold']].to_csv('results/test_predictions.csv')
    print("Test predictions saved to results/test_predictions.csv")
    
    # Generate and save plots
    # ROC Curve
    plt.figure()
    try:
        fpr, tpr, _ = roc_curve(y_test, all_probs)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['AUROC']:.3f})")
    except Exception:
        plt.plot([0, 1], [0, 1], label="Failed to generate ROC")
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    plt.close()
    
    # PR Curve
    plt.figure()
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, all_probs)
        plt.plot(recall_vals, precision_vals, label=f"PR Curve (AUC = {metrics['AUPRC']:.3f})")
    except Exception:
        plt.plot([0, 1], [0, 1], label="Failed to generate PR")
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig('results/pr_curve.png')
    plt.close()
    
    # Confusion Matrix
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, all_preds), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    print("Plots saved to results directory.")

if __name__ == "__main__":
    pass
