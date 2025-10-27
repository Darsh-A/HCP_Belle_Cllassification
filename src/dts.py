"""Decision Trees and Boosted DT with Random Forests"""

import pickle
import os
import json
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA

from utils import Utils


def load_best_params(model_type):
    """Load best parameters from JSON configuration file.
    
    Args:
        model_type: Either 'decision_tree' or 'random_forest'
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model_params.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config[model_type]


def train_decision_tree(
    data,
    tune_hyperparameters=False,
    params=None,
    param_grid=None,
    model_name='decision_tree_model'
):
    """
    Train a Decision Tree model on the provided data with optional hyperparameter tuning.
    
    Args:
        data: List of X_train, X_test, y_train, y_test
        tune_hyperparameters: If True, use GridSearchCV to find best hyperparameters (default=False)
        param_grid: Dictionary of hyperparameters to search. If None, uses default grid.
        
    Returns:
        dict: Dictionary with keys 'confusion_matrix', 'roc_auc_score', 'accuracy', 'model',
              'feature_importance', 'reduced_features', 'best_params' (if tuning enabled)
    """
    X, y, X_train, X_test, y_train, y_test = data
    
    if tune_hyperparameters:
        print("\n" + "="*80)
        print("DECISION TREE - HYPERPARAMETER TUNING")
        print("="*80)
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'criterion': ['gini', 'entropy']
            }
        
        print(f"Parameter grid: {param_grid}")
        print(f"Total combinations: {sum(1 for _ in GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid).get_params()['param_grid'])}")
        
        # Create base model
        base_model = DecisionTreeClassifier(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        print("\nStarting grid search...")
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\n✓ Grid search complete!")
        print(f"Best CV ROC-AUC: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        print("="*80)
    else:
        # Create model with default parameters from config
        if params is None:
            params = load_best_params('decision_tree')
        model = DecisionTreeClassifier(**params)
        best_params = None
    
    start_time = time.time()
    # Train model
    model.fit(X_train, y_train)
    end_time = time.time()
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    cm = Utils.give_conf_matrix(y_test, y_pred)
    roc_curve = Utils.give_roc_curve(y_test, y_prob)
    roc_auc_score = Utils.give_roc_auc_score(y_test, y_prob)
    accuracy = Utils.give_accuracy(y_test, y_pred)
    
    # Get feature importance
    feature_names = X.columns.tolist()
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    # Auto-calculate optimal threshold instead of using hardcoded value
    optimal_threshold = Utils.calculate_optimal_threshold(feature_importance)
    
    # Get reduced features based on auto-calculated threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= optimal_threshold]
    
    print(f"\nDecision Tree - Total features: {len(feature_names)}")

    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved Decision Tree model to {model_path}")
    
    # Return results as dictionary
    results = {
        'confusion_matrix': cm,
        'roc_curve': roc_curve,
        'roc_auc_score': roc_auc_score,
        'accuracy': accuracy,
        'model': model,
        'feature_importance': feature_importance,
        'reduced_features': reduced_features,
        'training_time': end_time - start_time
    }
    
    if tune_hyperparameters:
        results['best_params'] = best_params
        results['cv_results'] = grid_search.cv_results_
    
    return results


def train_random_forest(
        data, 
        params=None,
        tune_hyperparameters=False,
        param_grid=None,
        search_method='grid',
        model_name='random_forest_model'
    ):
    """
    Train a Random Forest model on the provided data with optional hyperparameter tuning.
    
    Args:
        data: List of X_train, X_test, y_train, y_test
        params: Manual parameters (ignored if tune_hyperparameters=True)
        tune_hyperparameters: If True, use GridSearchCV/RandomizedSearchCV (default=False)
        param_grid: Dictionary of hyperparameters to search. If None, uses default grid.
        search_method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV (default='grid')
        
    Returns:
        dict: Dictionary with keys 'confusion_matrix', 'roc_auc_score', 'accuracy', 'model',
              'feature_importance', 'reduced_features', 'best_params' (if tuning enabled)
    """

    X, y, X_train, X_test, y_train, y_test = data

    if tune_hyperparameters:
        print("\n" + "="*80)
        print("RANDOM FOREST - HYPERPARAMETER TUNING")
        print("="*80)
        
        # Default parameter grid if not provided
        if param_grid is None:
            if search_method == 'grid':
                # Smaller grid for GridSearchCV (faster)
                param_grid = {
                    'n_estimators': [100, 200, 400],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            else:
                # Larger parameter distributions for RandomizedSearchCV
                from scipy.stats import randint
                param_grid = {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [5, 10, 15, 20, 25, None],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
        
        print(f"Search method: {search_method.upper()}")
        print(f"Parameter grid: {param_grid}")
        
        # Create base model
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Choose search method
        if search_method == 'random':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=50,  # Number of parameter settings sampled
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1,
                random_state=42,
                return_train_score=True
            )
            print(f"Running randomized search with 50 iterations...")
        else:
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
            print(f"Running grid search...")
        
        print("\nStarting hyperparameter search...")
        search.fit(X_train, y_train)
        
        # Get best model
        model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"\n✓ Search complete!")
        print(f"Best CV ROC-AUC: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        print("="*80)
    else:
        # Use manual parameters or defaults from config
        if params is None:
            params = load_best_params('random_forest')
        
        # Create model
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 10),
            min_samples_leaf=params.get('min_samples_leaf', 5),
            random_state=params.get('random_state', 42),
            n_jobs=-1
        )
        best_params = None
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    cm = Utils.give_conf_matrix(y_test, y_pred)
    roc_curve = Utils.give_roc_curve(y_test, y_prob)
    roc_auc_score = Utils.give_roc_auc_score(y_test, y_prob)
    accuracy = Utils.give_accuracy(y_test, y_pred)
    
    # Get feature importance
    feature_names = X.columns.tolist()
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    # Auto-calculate optimal threshold instead of using hardcoded value
    optimal_threshold = Utils.calculate_optimal_threshold(feature_importance)
    
    # Get reduced features based on auto-calculated threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= optimal_threshold]
    
    print(f"\nRandom Forest - Total features: {len(feature_names)}")

    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved Random Forest model to {model_path}")
    
    # Return results as dictionary
    results = {
        'confusion_matrix': cm,
        'roc_curve': roc_curve,
        'roc_auc_score': roc_auc_score,
        'accuracy': accuracy,
        'model': model,
        'feature_importance': feature_importance,
        'reduced_features': reduced_features,
        'training_time': end_time - start_time
    }
    
    if tune_hyperparameters:
        results['best_params'] = best_params
        results['cv_results'] = search.cv_results_
    
    return results


if __name__ == "__main__":
    # Example usage
    df = Utils.data_import('data/data_hep - data_hep.csv')
    
    # Train Decision Tree with custom threshold
    dt_results = train_decision_tree(df, tune_hyperparameters=False)
    print("\nDecision Tree Results:")
    print(f"Accuracy: {dt_results['accuracy']:.4f}")
    print(f"ROC-AUC: {dt_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{dt_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(dt_results['reduced_features'])}")
    print(f"Reduced features: {dt_results['reduced_features']}")
    
    # Train Random Forest with custom threshold
    rf_results = train_random_forest(df, tune_hyperparameters=True)
    print("\nRandom Forest Results:")
    print(f"Accuracy: {rf_results['accuracy']:.4f}")
    print(f"ROC-AUC: {rf_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{rf_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(rf_results['reduced_features'])}")
    print(f"Reduced features: {rf_results['reduced_features']}")