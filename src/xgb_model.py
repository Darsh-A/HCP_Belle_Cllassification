"""XGBoost"""


import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from utils import Utils


def train_xgboost(data, importance_threshold=0.01, params=None):
    """
    Train an XGBoost model on the provided data.
    
    Args:
        data: pandas DataFrame containing the dataset
        importance_threshold: Threshold for feature importance (default=0.01)
        params: Optional dictionary of XGBoost parameters. If None, uses default parameters.
        
    Returns:
        dict: Dictionary with keys 'confusion_matrix', 'roc_auc_score', 'accuracy', 'model',
              'feature_importance', 'reduced_features'
    """
    # Prepare data
    X, y = Utils.bin_classification(data)

    X_train_full, X_test, y_train_full, y_test = Utils.data_split(X, y, ratio=0.3)
    X_train, X_val, y_train, y_val = Utils.data_split(X_train_full, y_train_full, ratio=0.2) # or use test_size=0.2
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())

    # Use default XGBoost hyperparameters if not provided
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc', 'error'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.5,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'device': 'cuda:1',
            'tree_method': 'hist'
        }
    
    # Train the model with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False  # Suppress training output
    )
    
    # Make predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    cm = Utils.give_conf_matrix(y_test, y_pred)
    roc_curve = Utils.give_roc_curve(y_test, y_pred_proba)
    roc_auc_score = Utils.give_roc_auc_score(y_test, y_pred_proba)
    accuracy = Utils.give_accuracy(y_test, y_pred)
    
    # Get feature importance (using 'gain' as importance type)
    importance_dict = model.get_score(importance_type='gain')
    
    # Normalize feature importance to sum to 1 for threshold comparison
    total_importance = sum(importance_dict.values())
    feature_importance = {k: v / total_importance for k, v in importance_dict.items()}
    
    # Get reduced features based on threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= importance_threshold]
    
    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "xgboost_model.json")
    model.save_model(model_path)
    
    # Return results as dictionary
    return {
        'confusion_matrix': cm,
        'roc_auc_score': roc_auc_score,
        'roc_curve': roc_curve,
        'accuracy': accuracy,
        'model': model,
        'feature_importance': feature_importance,
        'reduced_features': reduced_features
    }


if __name__ == "__main__":
    # Example usage
    df = Utils.data_import('data/data_hep - data_hep.csv')
    
    # Train XGBoost with custom threshold
    xgb_results = train_xgboost(df, importance_threshold=0.01)
    print("\nXGBoost Results:")
    print(f"Accuracy: {xgb_results['accuracy']:.4f}")
    print(f"ROC-AUC: {xgb_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{xgb_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(xgb_results['reduced_features'])}")
    print(f"Reduced features: {xgb_results['reduced_features']}")
