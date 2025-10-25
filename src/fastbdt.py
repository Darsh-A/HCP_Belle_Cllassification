"""FastBDT Model"""

import pandas as pd
import numpy as np
import pickle
import os
from PyFastBDT import FastBDT
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

from utils import Utils


def train_fastbdt(
    data,
    importance_threshold=0.01,
    nTrees = 150,
    depth = 6,
    shrinkage = 0.1,
    subsample = 0.8,
    transform2probability=True,
    binning=None
):
    """
    Train a FastBDT model on the provided data.
    
    Args:
        data: pandas DataFrame containing the dataset
        importance_threshold: Threshold for feature importance (default=0.01)
        
    Returns:
        dict: Dictionary with keys 'confusion_matrix', 'roc_auc_score', 'accuracy', 'model',
              'feature_importance', 'reduced_features'
    """
    # Prepare data
    X, y = Utils.bin_classification(data)
    
    X_train, X_test, y_train, y_test = Utils.data_split(X, y, ratio=0.3)
    
    bdt = FastBDT.Classifier(
        nTrees = nTrees,
        depth = depth,
        shrinkage = shrinkage,
        subsample = subsample,
        transform2probability=transform2probability,
        binning = binning if binning is not None else []
    )
    bdt.fit(X_train.values, y_train.values)
    
    # Make predictions
    y_pred_scores = bdt.predict(X_test.values)
    y_pred_class = (y_pred_scores > 0.5).astype(int)
    
    # Calculate metrics
    cm = Utils.give_conf_matrix(y_test, y_pred_class)
    roc_curve = Utils.give_roc_curve(y_test, y_pred_scores)
    roc_auc_score = Utils.give_roc_auc_score(y_test, y_pred_scores)
    accuracy = Utils.give_accuracy(y_test, y_pred_class)
    
    # Get feature importance
    feature_importance_scores = bdt.internFeatureImportance()
    feature_names = X.columns.tolist()
    
    # Create feature importance dictionary
    # Note: FastBDT returns importance scores, normalize them
    total_importance = sum(feature_importance_scores)
    feature_importance = {
        feature_names[i]: feature_importance_scores[i] / total_importance 
        for i in range(len(feature_names))
    }
    
    # Get reduced features based on threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= importance_threshold]
    
    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "fastbdt_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(bdt, f)
    
    # Return results as dictionary
    return {
        'confusion_matrix': cm,
        'roc_curve': roc_curve,
        'roc_auc_score': roc_auc_score,
        'accuracy': accuracy,
        'model': bdt,
        'feature_importance': feature_importance,
        'reduced_features': reduced_features
    }


if __name__ == "__main__":
    # Example usage
    df = Utils.data_import('data/data_hep - data_hep.csv')
    
    # Train FastBDT with custom threshold
    fastbdt_results = train_fastbdt(df, importance_threshold=0.01)
    print("\nFastBDT Results:")
    print(f"Accuracy: {fastbdt_results['accuracy']:.4f}")
    print(f"ROC-AUC: {fastbdt_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{fastbdt_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(fastbdt_results['reduced_features'])}")
    print(f"Reduced features: {fastbdt_results['reduced_features']}")
