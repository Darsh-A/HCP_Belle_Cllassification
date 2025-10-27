"""FastBDT Model"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import random
import time
from PyFastBDT import FastBDT
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score

from utils import Utils




def load_best_params():
    """Load best parameters from JSON configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model_params.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['fastbdt']


def randomized_search_fastbdt(
    data,
    param_distributions=None,
    n_iter=20,
    n_folds=5,
    random_state=42
):
    """
    Perform randomized search for FastBDT hyperparameter tuning.
    
    Args:
        data: Tuple of (X, y, X_train, X_test, y_train, y_test)
        param_distributions: Dictionary of parameter distributions to sample from.
                           Each value can be a list (discrete choices) or tuple (min, max) for continuous.
        n_iter: Number of random parameter combinations to try (default=20)
        n_folds: Number of cross-validation folds (default=5)
        random_state: Random seed for reproducibility (default=42)
        
    Returns:
        dict: Dictionary with 'best_params', 'best_score', 'cv_results'
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Extract data from tuple
    X, y, _, _, _, _ = data
    
    # Default parameter distributions if not provided
    if param_distributions is None:
        param_distributions = {
            'nTrees': [50, 100, 150, 200, 250, 300],
            'depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'shrinkage': (0.01, 0.3),  # Continuous range (learning rate)
            'subsample': (0.5, 1.0),  # Continuous range
        }
    
    best_score = -np.inf
    best_params = None
    cv_results = []
    
    print(f"Starting Randomized Search with {n_iter} iterations...\n")
    
    # Setup K-Fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for i in range(n_iter):
        # Sample random parameters
        current_params = {}
        
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
        if 'shrinkage' in current_params:
            current_params['shrinkage'] = round(current_params['shrinkage'], 4)
        if 'subsample' in current_params:
            current_params['subsample'] = round(current_params['subsample'], 4)
        
        # Perform cross-validation
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train FastBDT with current parameters
            bdt = FastBDT.Classifier(
                nTrees=current_params.get('nTrees', 150),
                depth=current_params.get('depth', 6),
                shrinkage=current_params.get('shrinkage', 0.1),
                subsample=current_params.get('subsample', 0.8),
                transform2probability=True
            )
            
            bdt.fit(X_train_fold.values, y_train_fold.values)
            
            # Predict and calculate AUC
            y_pred_proba = bdt.predict(X_val_fold.values)
            fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
            fold_scores.append(fold_auc)
        
        # Calculate mean and std of AUC across folds
        mean_auc = np.mean(fold_scores)
        std_auc = np.std(fold_scores)
        
        # Store result
        result_entry = {
            'params': current_params.copy(),
            'mean_auc': mean_auc,
            'std_auc': std_auc
        }
        cv_results.append(result_entry)
        
        # Update best parameters
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = current_params.copy()
        
        print(f"Iteration {i+1}/{n_iter}: AUC = {mean_auc:.6f} (+/- {std_auc:.6f})")
        print(f"  Params: nTrees={current_params.get('nTrees')}, "
              f"depth={current_params.get('depth')}, "
              f"shrinkage={current_params.get('shrinkage')}, "
              f"subsample={current_params.get('subsample')}")
    
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




def load_best_params():
    """Load best parameters from JSON configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model_params.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['fastbdt']


def randomized_search_fastbdt(
    data,
    param_distributions=None,
    n_iter=20,
    n_folds=5,
    random_state=42
):
    """
    Perform randomized search for FastBDT hyperparameter tuning.
    
    Args:
        data: Tuple of (X, y, X_train, X_test, y_train, y_test)
        param_distributions: Dictionary of parameter distributions to sample from.
                           Each value can be a list (discrete choices) or tuple (min, max) for continuous.
        n_iter: Number of random parameter combinations to try (default=20)
        n_folds: Number of cross-validation folds (default=5)
        random_state: Random seed for reproducibility (default=42)
        
    Returns:
        dict: Dictionary with 'best_params', 'best_score', 'cv_results'
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Extract data from tuple
    X, y, _, _, _, _ = data
    
    # Default parameter distributions if not provided
    if param_distributions is None:
        param_distributions = {
            'nTrees': [50, 100, 150, 200, 250, 300],
            'depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'shrinkage': (0.01, 0.3),  # Continuous range (learning rate)
            'subsample': (0.5, 1.0),  # Continuous range
        }
    
    best_score = -np.inf
    best_params = None
    cv_results = []
    
    print(f"Starting Randomized Search with {n_iter} iterations...\n")
    
    # Setup K-Fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for i in range(n_iter):
        # Sample random parameters
        current_params = {}
        
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
        if 'shrinkage' in current_params:
            current_params['shrinkage'] = round(current_params['shrinkage'], 4)
        if 'subsample' in current_params:
            current_params['subsample'] = round(current_params['subsample'], 4)
        
        # Perform cross-validation
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train FastBDT with current parameters
            bdt = FastBDT.Classifier(
                nTrees=current_params.get('nTrees', 150),
                depth=current_params.get('depth', 6),
                shrinkage=current_params.get('shrinkage', 0.1),
                subsample=current_params.get('subsample', 0.8),
                transform2probability=True
            )
            
            bdt.fit(X_train_fold.values, y_train_fold.values)
            
            # Predict and calculate AUC
            y_pred_proba = bdt.predict(X_val_fold.values)
            fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
            fold_scores.append(fold_auc)
        
        # Calculate mean and std of AUC across folds
        mean_auc = np.mean(fold_scores)
        std_auc = np.std(fold_scores)
        
        # Store result
        result_entry = {
            'params': current_params.copy(),
            'mean_auc': mean_auc,
            'std_auc': std_auc
        }
        cv_results.append(result_entry)
        
        # Update best parameters
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = current_params.copy()
        
        print(f"Iteration {i+1}/{n_iter}: AUC = {mean_auc:.6f} (+/- {std_auc:.6f})")
        print(f"  Params: nTrees={current_params.get('nTrees')}, "
              f"depth={current_params.get('depth')}, "
              f"shrinkage={current_params.get('shrinkage')}, "
              f"subsample={current_params.get('subsample')}")
    
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


def train_fastbdt(
    data,
    nTrees=None,
    depth=None,
    shrinkage=None,
    subsample=None,
    transform2probability=None,
    binning=None,
    tune_hyperparameters=False,
    param_distributions=None,
    n_iter=20,
    n_folds=5,
    random_state=42,
    model_name='fastbdt_model'
):
    """
    Train a FastBDT model on the provided data.
    
    Args:
        data: Tuple of (X, y, X_train, X_test, y_train, y_test)
        nTrees: Number of trees (default=None, loads from config)
        depth: Maximum tree depth (default=None, loads from config)
        shrinkage: Learning rate (default=None, loads from config)
        subsample: Subsample ratio (default=None, loads from config)
        transform2probability: Transform to probability (default=None, loads from config)
        binning: Custom binning list (default=None)
        tune_hyperparameters: If True, performs randomized search for hyperparameter tuning (default=False)
        param_distributions: Parameter distributions for randomized search (only used if tune_hyperparameters=True)
        n_iter: Number of iterations for randomized search (default=20)
        n_folds: Number of CV folds for tuning (default=5)
        random_state: Random seed for reproducibility (default=42)
        
    Returns:
        dict: Dictionary with keys 'confusion_matrix', 'roc_auc_score', 'accuracy', 'model',
              'feature_importance', 'reduced_features', and optionally 'tuning_results' if tuning was performed
    """
    # Load best parameters from config if not provided
    best_params = load_best_params()
    if nTrees is None:
        nTrees = best_params['nTrees']
    if depth is None:
        depth = best_params['depth']
    if shrinkage is None:
        shrinkage = best_params['shrinkage']
    if subsample is None:
        subsample = best_params['subsample']
    if transform2probability is None:
        transform2probability = best_params['transform2probability']
    
    # Perform hyperparameter tuning if requested
    tuning_results = None
    if tune_hyperparameters:
        print("Starting hyperparameter tuning with Randomized Search...")
        tuning_results = randomized_search_fastbdt(
            data=data,
            param_distributions=param_distributions,
            n_iter=n_iter,
            n_folds=n_folds,
            random_state=random_state
        )
        # Use best parameters from tuning
        best_params_tuned = tuning_results['best_params']
        nTrees = best_params_tuned.get('nTrees', nTrees)
        depth = best_params_tuned.get('depth', depth)
        shrinkage = best_params_tuned.get('shrinkage', shrinkage)
        subsample = best_params_tuned.get('subsample', subsample)
        print(f"Tuning complete. Using best parameters for final model training.\n")
    
    # Prepare data
    X, y, X_train, X_test, y_train, y_test = data
    
    bdt = FastBDT.Classifier(
        nTrees=nTrees,
        depth=depth,
        shrinkage=shrinkage,
        subsample=subsample,
        transform2probability=transform2probability,
        binning=binning if binning is not None else []
    )
    start_time = time.time()
    bdt.fit(X_train.values, y_train.values)
    end_time = time.time()
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
    
    # Auto-calculate optimal threshold instead of using hardcoded value
    optimal_threshold = Utils.calculate_optimal_threshold(feature_importance)
    
    # Get reduced features based on auto-calculated threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= optimal_threshold]
    
    print(f"\nFastBDT - Total features: {len(feature_names)}")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(bdt, f)
    
    # Return results as dictionary
    result = {
        'confusion_matrix': cm,
        'roc_curve': roc_curve,
        'roc_auc_score': roc_auc_score,
        'accuracy': accuracy,
        'model': bdt,
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
    
    # Example 1: Train FastBDT without tuning (default behavior)
    print("=" * 80)
    print("Training FastBDT WITHOUT hyperparameter tuning...")
    print("=" * 80)
    fastbdt_results = train_fastbdt(df, tune_hyperparameters=False)
    print("\nFastBDT Results (No Tuning):")
    print(f"Accuracy: {fastbdt_results['accuracy']:.4f}")
    print(f"ROC-AUC: {fastbdt_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{fastbdt_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(fastbdt_results['reduced_features'])}")
    print(f"Reduced features: {fastbdt_results['reduced_features']}")
    
    # Example 2: Train FastBDT WITH hyperparameter tuning (toggle ON)
    print("\n" + "=" * 80)
    print("Training FastBDT WITH hyperparameter tuning (Randomized Search)...")
    print("=" * 80)
    fastbdt_results_tuned = train_fastbdt(
        df,
        tune_hyperparameters=True,  # Toggle ON
        n_iter=10,  # Number of random parameter combinations to try
        n_folds=5,  # Number of CV folds
        random_state=42
    )
    print("\nFastBDT Results (With Tuning):")
    print(f"Accuracy: {fastbdt_results_tuned['accuracy']:.4f}")
    print(f"ROC-AUC: {fastbdt_results_tuned['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{fastbdt_results_tuned['confusion_matrix']}")
    print(f"Number of reduced features: {len(fastbdt_results_tuned['reduced_features'])}")
    print(f"Best tuned parameters: {fastbdt_results_tuned['tuning_results']['best_params']}")