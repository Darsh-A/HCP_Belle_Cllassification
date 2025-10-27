"""XGBoost"""


import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from utils import Utils


def load_best_params():
    """Load best parameters from JSON configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model_params.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['xgboost']


def randomized_search_xgboost(
    data, 
    param_distributions=None, 
    n_iter=20, 
    n_folds=5, 
    num_boost_round=500,
    early_stopping_rounds=50,
    random_state=42
):
    """
    Perform randomized search for XGBoost hyperparameter tuning.
    
    Args:
        data: pandas DataFrame containing the dataset
        param_distributions: Dictionary of parameter distributions to sample from.
                           Each value can be a list (discrete choices) or tuple (min, max) for continuous.
        n_iter: Number of random parameter combinations to try (default=20)
        n_folds: Number of cross-validation folds (default=5)
        num_boost_round: Maximum number of boosting rounds (default=500)
        early_stopping_rounds: Early stopping rounds for CV (default=50)
        random_state: Random seed for reproducibility (default=42)
        
    Returns:
        dict: Dictionary with 'best_params', 'best_score', 'cv_results'
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Prepare data
    X, y, X_train, X_test, y_train, y_test = data
    
    # Create DMatrix for CV
    dtrain = xgb.DMatrix(X, label=y, feature_names=X.columns.tolist())
    
    # Default parameter distributions if not provided
    if param_distributions is None:
        param_distributions = {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': (0.01, 0.3),  # Continuous range
            'subsample': (0.5, 1.0),  # Continuous range
            'colsample_bytree': (0.5, 1.0),  # Continuous range
            'min_child_weight': [1, 3, 5, 7, 10],
            'gamma': [0, 0.1, 0.2, 0.3, 0.5],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0, 5.0]
        }
    
    # Fixed parameters
    fixed_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device': 'cuda',
        'tree_method': 'hist'
    }
    
    best_score = -np.inf
    best_params = None
    cv_results = []
    
    print(f"Starting Randomized Search with {n_iter} iterations...\n")
    
    for i in range(n_iter):
        # Sample random parameters
        current_params = fixed_params.copy()
        
        for param_name, param_range in param_distributions.items():
            if isinstance(param_range, (list, tuple)) and len(param_range) == 2 and not isinstance(param_range, list):
                # Continuous range (tuple)
                if isinstance(param_range[0], float) or isinstance(param_range[1], float):
                    current_params[param_name] = np.random.uniform(param_range[0], param_range[1])
                else:
                    current_params[param_name] = random.randint(param_range[0], param_range[1])
            else:
                # Discrete choices (list)
                current_params[param_name] = random.choice(param_range)
        
        # Round learning_rate for cleaner output
        if 'learning_rate' in current_params:
            current_params['learning_rate'] = round(current_params['learning_rate'], 4)
        if 'subsample' in current_params:
            current_params['subsample'] = round(current_params['subsample'], 4)
        if 'colsample_bytree' in current_params:
            current_params['colsample_bytree'] = round(current_params['colsample_bytree'], 4)
        
        # Perform cross-validation
        cv_result = xgb.cv(
            current_params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=n_folds,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            seed=random_state
        )
        
        # Get best score (last row, test-auc-mean)
        mean_auc = cv_result['test-auc-mean'].iloc[-1]
        std_auc = cv_result['test-auc-std'].iloc[-1]
        n_estimators = len(cv_result)
        
        # Store result
        result_entry = {
            'params': current_params.copy(),
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'n_estimators': n_estimators
        }
        cv_results.append(result_entry)
        
        # Update best parameters
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = current_params.copy()
            best_params['n_estimators'] = n_estimators
        
        print(f"Iteration {i+1}/{n_iter}: AUC = {mean_auc:.6f} (+/- {std_auc:.6f}), n_estimators = {n_estimators}")
        print(f"  Params: max_depth={current_params.get('max_depth')}, "
              f"lr={current_params.get('learning_rate')}, "
              f"subsample={current_params.get('subsample')}, "
              f"colsample_bytree={current_params.get('colsample_bytree')}")
    
    print(f"\n{'='*80}")
    print(f"Best Score (AUC): {best_score:.6f}")
    print(f"Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': cv_results
    }


def train_xgboost(
    data,
    params=None, 
    tune_hyperparameters=False,
    param_distributions=None,
    n_iter=20,
    n_folds=5,
    random_state=42,
    model_name='xgboost_model'
):
    """
    Train an XGBoost model on the provided data.
    
    Args:
        data: pandas DataFrame containing the dataset
        params: Optional dictionary of XGBoost parameters. If None, uses default parameters.
        tune_hyperparameters: If True, performs randomized search for hyperparameter tuning (default=False)
        param_distributions: Parameter distributions for randomized search (only used if tune_hyperparameters=True)
        n_iter: Number of iterations for randomized search (default=20)
        n_folds: Number of CV folds for tuning (default=5)
        random_state: Random seed for reproducibility (default=42)
        
    Returns:
        dict: Dictionary with keys 'confusion_matrix', 'roc_auc_score', 'accuracy', 'model',
              'feature_importance', 'reduced_features', and optionally 'tuning_results' if tuning was performed
    """
    # Perform hyperparameter tuning if requested
    tuning_results = None
    if tune_hyperparameters:
        print("Starting hyperparameter tuning with Randomized Search...")
        tuning_results = randomized_search_xgboost(
            data=data,
            param_distributions=param_distributions,
            n_iter=n_iter,
            n_folds=n_folds,
            random_state=random_state
        )
        # Use best parameters from tuning
        params = tuning_results['best_params']
        print(f"Tuning complete. Using best parameters for final model training.\n")
    
    # Prepare data
    X, y, X_train, X_test, y_train, y_test = data

    X_train_full, X_test, y_train_full, y_test = Utils.data_split(X, y, ratio=0.3)
    X_train, X_val, y_train, y_val = Utils.data_split(X_train_full, y_train_full, ratio=0.2) # or use test_size=0.2
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())

    # Use default XGBoost hyperparameters if not provided
    if params is None:
        params = load_best_params()
    
    # Determine num_boost_round
    num_boost_round = params.pop('n_estimators', 500)
    
    # Train the model with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    start_time = time.time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False  # Suppress training output
    )
    end_time = time.time()
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
    
    # Auto-calculate optimal threshold instead of using hardcoded value
    optimal_threshold = Utils.calculate_optimal_threshold(feature_importance)
    
    # Get reduced features based on auto-calculated threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= optimal_threshold]
    
    print(f"\nXGBoost - Total features: {len(X.columns)}")
    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.json")
    model.save_model(model_path)
    
    # Return results as dictionary
    result = {
        'confusion_matrix': cm,
        'roc_auc_score': roc_auc_score,
        'roc_curve': roc_curve,
        'accuracy': accuracy,
        'model': model,
        'feature_importance': feature_importance,
        'reduced_features': reduced_features,
        'training_time': end_time - start_time
    }
    
    # Add tuning results if hyperparameter tuning was performed
    if tuning_results is not None:
        result['tuning_results'] = tuning_results
    
    return result


if __name__ == "__main__":
    # Example usage
    df = Utils.data_import('data/data_hep - data_hep.csv')
    
    # Example 1: Train XGBoost without tuning (default behavior)
    print("=" * 80)
    print("Training XGBoost WITHOUT hyperparameter tuning...")
    print("=" * 80)
    xgb_results = train_xgboost(df, tune_hyperparameters=False)
    print("\nXGBoost Results (No Tuning):")
    print(f"Accuracy: {xgb_results['accuracy']:.4f}")
    print(f"ROC-AUC: {xgb_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{xgb_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(xgb_results['reduced_features'])}")
    print(f"Reduced features: {xgb_results['reduced_features']}")
    
    # Example 2: Train XGBoost WITH hyperparameter tuning (toggle ON)
    print("\n" + "=" * 80)
    print("Training XGBoost WITH hyperparameter tuning (Randomized Search)...")
    print("=" * 80)
    xgb_results_tuned = train_xgboost(
        df, 
        tune_hyperparameters=True,  # Toggle ON
        n_iter=10,  # Number of random parameter combinations to try
        n_folds=5,  # Number of CV folds
        random_state=42
    )
    print("\nXGBoost Results (With Tuning):")
    print(f"Accuracy: {xgb_results_tuned['accuracy']:.4f}")
    print(f"ROC-AUC: {xgb_results_tuned['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{xgb_results_tuned['confusion_matrix']}")
    print(f"Number of reduced features: {len(xgb_results_tuned['reduced_features'])}")
    print(f"Best tuned parameters: {xgb_results_tuned['tuning_results']['best_params']}")