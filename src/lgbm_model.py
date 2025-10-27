"""LightGBM Model"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import random
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from utils import Utils


def load_best_params():
    """Load best parameters from JSON configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model_params.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['lightgbm']


def _check_cuda_availability():
    """
    Check if CUDA is available for LightGBM.
    
    Returns:
        str: 'cuda' if available, 'cpu' otherwise
    """
    try:
        # Try to create a simple dataset and train with CUDA
        test_data = lgb.Dataset(np.random.rand(10, 5), label=np.random.randint(0, 2, 10))
        test_params = {'objective': 'binary', 'device': 'cuda', 'verbosity': -1}
        lgb.train(test_params, test_data, num_boost_round=1)
        device = 'cuda'
        print("✅ CUDA is available - LightGBM will use GPU")
    except Exception:
        device = 'cpu'
        print("⚠️  CUDA not available - LightGBM will use CPU")
    
    return device


def randomized_search_lightgbm(
    X_train, 
    y_train,
    param_distributions=None, 
    n_iter=20, 
    n_folds=5, 
    num_boost_round=1500,
    early_stopping_rounds=50,
    random_state=42
):
    """
    Perform randomized search for LightGBM hyperparameter tuning.
    
    Args:
        X_train: Training features (only training data, not full dataset!)
        y_train: Training labels (only training data, not full dataset!)
        param_distributions: Dictionary of parameter distributions to sample from.
                           Each value can be a list (discrete choices) or tuple (min, max) for continuous.
        n_iter: Number of random parameter combinations to try (default=20)
        n_folds: Number of cross-validation folds (default=5)
        num_boost_round: Maximum number of boosting rounds (default=1500)
        early_stopping_rounds: Early stopping rounds for CV (default=50)
        random_state: Random seed for reproducibility (default=42)
        
    Returns:
        dict: Dictionary with 'best_params', 'best_score', 'cv_results'
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Create LightGBM Dataset for CV - ONLY from training data to prevent data leakage
    dtrain = lgb.Dataset(X_train, label=y_train)
    
    # Default parameter distributions if not provided
    if param_distributions is None:
        param_distributions = {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, -1],  # -1 means no limit
            'learning_rate': (0.01, 0.3),  # Continuous range
            'subsample': (0.5, 1.0),  # Continuous range
            'colsample_bytree': (0.5, 1.0),  # Continuous range
            'min_child_samples': [5, 10, 20, 30, 50],
            'num_leaves': [15, 31, 63, 127, 255],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
            'min_split_gain': [0, 0.01, 0.1, 0.5, 1.0]
        }
    
    # Detect CUDA availability
    device = _check_cuda_availability()
    
    # Fixed parameters
    fixed_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'device': device
    }
    
    best_score = -np.inf
    best_params = None
    cv_results = []
    
    print(f"Starting Randomized Search with {n_iter} iterations...\n")
    
    for i in range(n_iter):
        # Sample random parameters
        current_params = fixed_params.copy()
        
        for param_name, param_range in param_distributions.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Continuous range (tuple)
                if isinstance(param_range[0], float) or isinstance(param_range[1], float):
                    current_params[param_name] = np.random.uniform(param_range[0], param_range[1])
                else:
                    current_params[param_name] = random.randint(param_range[0], param_range[1])
            else:
                # Discrete choices (list)
                current_params[param_name] = random.choice(param_range)
        
        # Round continuous parameters for cleaner output
        if 'learning_rate' in current_params:
            current_params['learning_rate'] = round(current_params['learning_rate'], 4)
        if 'subsample' in current_params:
            current_params['subsample'] = round(current_params['subsample'], 4)
        if 'colsample_bytree' in current_params:
            current_params['colsample_bytree'] = round(current_params['colsample_bytree'], 4)
        
        # Perform cross-validation
        cv_result = lgb.cv(
            current_params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=n_folds,
            stratified=True,
            shuffle=True,
            seed=random_state,
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
        )
        
        # Get best score (mean AUC from last iteration)
        mean_auc = cv_result['valid auc-mean'][-1]
        std_auc = cv_result['valid auc-stdv'][-1]
        n_estimators = len(cv_result['valid auc-mean'])
        
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


def train_lightgbm(
    data,
    params=None, 
    tune_hyperparameters=False,
    param_distributions=None,
    n_iter=20,
    n_folds=5,
    random_state=42,
    model_name='lightgbm_model'
):
    """
    Train a LightGBM model on the provided data.
    
    Args:
        data: Tuple of (X, y, X_train, X_test, y_train, y_test)
              The full X and y are ignored to follow the Golden Rule.
              Only the pre-split X_train, X_test, y_train, y_test are used.
        params: Optional dictionary of LightGBM parameters. If None, uses default parameters.
        tune_hyperparameters: If True, performs randomized search for hyperparameter tuning (default=False)
        param_distributions: Parameter distributions for randomized search (only used if tune_hyperparameters=True)
        n_iter: Number of iterations for randomized search (default=20)
        n_folds: Number of CV folds for tuning (default=5)
        random_state: Random seed for reproducibility (default=42)
        
    Returns:
        dict: Dictionary with keys 'confusion_matrix', 'roc_auc_score', 'accuracy', 'model',
              'feature_importance', 'reduced_features', and optionally 'tuning_results' if tuning was performed
    """

    # We unpack everything, but we consciously only use the pre-split data.
    # The full X and y are ignored here to follow the Golden Rule.
    _, _, X_train_full, X_test, y_train_full, y_test = data
    
    # Perform hyperparameter tuning if requested
    tuning_results = None
    if tune_hyperparameters:
        print("Starting hyperparameter tuning with Randomized Search...")
        print("Using ONLY training data for CV to prevent data leakage...")
        # --- CRITICAL FIX: Pass ONLY the training data to the tuning function ---
        tuning_results = randomized_search_lightgbm(
            X_train=X_train_full,  # Use the pre-split training set
            y_train=y_train_full,  # Use the pre-split training set
            param_distributions=param_distributions,
            n_iter=n_iter,
            n_folds=n_folds,
            random_state=random_state
        )
        # Use best parameters from tuning
        params = tuning_results['best_params']
        print(f"Tuning complete. Using best parameters for final model training.\n")
    
    # This split is for early stopping ONLY. It takes a small piece from the
    # training set, leaving the original test set untouched and pristine.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    # Create LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    # Detect CUDA availability if not tuning (tuning already sets device)
    if not tune_hyperparameters:
        device = _check_cuda_availability()
    else:
        # If tuning was done, device was already set in best_params
        device = params.get('device', 'cpu')
    
    # Use default LightGBM hyperparameters if not provided
    if params is None:
        params = load_best_params()
        params['device'] = device
    elif 'device' not in params:
        # If params provided but no device specified, set it
        params['device'] = device
    
    # Determine num_boost_round
    num_boost_round = params.pop('n_estimators', 500)
    
    # Train the model with early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0)  # Suppress training output
    ]
    
    start_time = time.time()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    end_time = time.time()
    # Make predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    cm = Utils.give_conf_matrix(y_test, y_pred)
    roc_curve = Utils.give_roc_curve(y_test, y_pred_proba)
    roc_auc_score = Utils.give_roc_auc_score(y_test, y_pred_proba)
    accuracy = Utils.give_accuracy(y_test, y_pred)
    
    # Get feature importance (using 'gain' as importance type)
    feature_names = X_train.columns.tolist()
    importance_scores = model.feature_importance(importance_type='gain')
    
    # Normalize feature importance to sum to 1 for threshold comparison
    total_importance = sum(importance_scores)
    feature_importance = {
        feature_names[i]: importance_scores[i] / total_importance 
        for i in range(len(feature_names))
    }
    
    # Auto-calculate optimal threshold instead of using hardcoded value
    optimal_threshold = Utils.calculate_optimal_threshold(feature_importance)
    
    # Get reduced features based on auto-calculated threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= optimal_threshold]
    
    print(f"\nLightGBM - Total features: {len(feature_names)}")
    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.txt")
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
    # Example usage with PROPER data splitting (no data leakage!)
    df = Utils.data_import('data/data_hep - data_hep.csv')
    
    # Get full X and y
    X, y = Utils.bin_classification(df)
    
    # ---- THE ONE AND ONLY DATA SPLIT ----
    # This is your single source of truth for training and testing.
    X_train, X_test, y_train, y_test = Utils.data_split(X, y, ratio=0.3, random_state=42)
    # ------------------------------------
    
    # Assemble the data package exactly as your other models expect it
    data_package = (X, y, X_train, X_test, y_train, y_test)
    
    # Example 1: Train LightGBM without tuning (default behavior)
    print("=" * 80)
    print("Training LightGBM WITHOUT hyperparameter tuning...")
    print("=" * 80)
    lgbm_results = train_lightgbm(data_package, importance_threshold=0.01, tune_hyperparameters=False)
    print("\nLightGBM Results (No Tuning):")
    print(f"Accuracy: {lgbm_results['accuracy']:.4f}")
    print(f"ROC-AUC: {lgbm_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{lgbm_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(lgbm_results['reduced_features'])}")
    print(f"Reduced features: {lgbm_results['reduced_features']}")
    
    # Example 2: Train LightGBM WITH hyperparameter tuning (toggle ON)
    print("\n" + "=" * 80)
    print("Training LightGBM WITH hyperparameter tuning (Randomized Search)...")
    print("=" * 80)
    lgbm_results_tuned = train_lightgbm(
        data_package,  # Use the SAME data package for consistent results!
        importance_threshold=0.01, 
        tune_hyperparameters=True,  # Toggle ON
        n_iter=20,  # Using 20 iterations for a more robust search
        n_folds=5,  # Number of CV folds
        random_state=42
    )
    print("\nLightGBM Results (With Tuning):")
    print(f"Accuracy: {lgbm_results_tuned['accuracy']:.4f}")
    print(f"ROC-AUC: {lgbm_results_tuned['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{lgbm_results_tuned['confusion_matrix']}")
    print(f"Number of reduced features: {len(lgbm_results_tuned['reduced_features'])}")
    print(f"Best tuned parameters: {lgbm_results_tuned['tuning_results']['best_params']}")
