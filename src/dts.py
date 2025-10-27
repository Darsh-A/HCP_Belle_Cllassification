"""Decision Trees and Boosted DT with Random Forests"""

import pickle
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


from utils import Utils


def train_decision_tree(data, importance_threshold=0.01):
    """
    Train a Decision Tree model on the provided data.
    
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
    
    # Create model
    model = DecisionTreeClassifier(random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
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
    
    # Get reduced features based on threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= importance_threshold]
    
    print(f"\nDecision Tree - Total features: {len(feature_names)}")
    print(f"Decision Tree - Reduced features (importance >= {importance_threshold}): {len(reduced_features)}")
    
    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "decision_tree_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved Decision Tree model to {model_path}")
    
    # Return results as dictionary
    return {
        'confusion_matrix': cm,
        'roc_curve': roc_curve,
        'roc_auc_score': roc_auc_score,
        'accuracy': accuracy,
        'model': model,
        'feature_importance': feature_importance,
        'reduced_features': reduced_features
    }


def train_random_forest(data, importance_threshold=0.01):
    """
    Train a Random Forest model on the provided data.
    
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
    
    # Create model
    model = RandomForestClassifier(random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
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
    
    # Get reduced features based on threshold
    reduced_features = [feature for feature, importance in feature_importance.items() 
                       if importance >= importance_threshold]
    
    print(f"\nRandom Forest - Total features: {len(feature_names)}")
    print(f"Random Forest - Reduced features (importance >= {importance_threshold}): {len(reduced_features)}")
    
    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "random_forest_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved Random Forest model to {model_path}")
    
    # Return results as dictionary
    return {
        'confusion_matrix': cm,
        'roc_curve': roc_curve,
        'roc_auc_score': roc_auc_score,
        'accuracy': accuracy,
        'model': model,
        'feature_importance': feature_importance,
        'reduced_features': reduced_features
    }


if __name__ == "__main__":
    # Example usage
    df = Utils.data_import('data/data_hep - data_hep.csv')
    
    # Train Decision Tree with custom threshold
    dt_results = train_decision_tree(df, importance_threshold=0.01)
    print("\nDecision Tree Results:")
    print(f"Accuracy: {dt_results['accuracy']:.4f}")
    print(f"ROC-AUC: {dt_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{dt_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(dt_results['reduced_features'])}")
    print(f"Reduced features: {dt_results['reduced_features']}")
    
    # Train Random Forest with custom threshold
    rf_results = train_random_forest(df, importance_threshold=0.01)
    print("\nRandom Forest Results:")
    print(f"Accuracy: {rf_results['accuracy']:.4f}")
    print(f"ROC-AUC: {rf_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{rf_results['confusion_matrix']}")
    print(f"Number of reduced features: {len(rf_results['reduced_features'])}")
    print(f"Reduced features: {rf_results['reduced_features']}")
