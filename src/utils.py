"""
Any Misc function goes here
like decorations, utilities, etc.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Utils:

    @staticmethod
    def data_import(data):
        """Import CSV data into a pandas DataFrame"""
        df = pd.read_csv(data)
        df.columns = df.columns.str.strip() # Strip whitespace from column names
        return df
    
    @staticmethod
    def data_split(x,y,ratio=0.3):
        X_train, X_test, y_train, y_test = train_test_split(
             x, y, test_size=ratio, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def bin_classification(df):
        """Binary classification: 0,1 = signal; 2,3,4,5 = background"""
        df['is_signal'] = df['type'].isin([0, 1]).astype(int)
        y = df['is_signal']
        features_to_drop = ['index', 'type', 'is_signal']
        X = df.drop(columns=features_to_drop)
        return X, y
    
    @staticmethod
    def plot_roc(y_test, y_pred_scores,plotname):

        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)


        fpr, tpr, thresholds = roc_curve(y_test, y_pred_scores)
        roc_auc = auc(fpr, tpr)

        print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        print("\nGenerating ROC curve plot...")
        plot_path = os.path.join(plots_dir, plotname)
        plt.savefig(plot_path)
        print(f"Plot saved as '{plot_path}'")
        plt.show()

        return roc_auc
    
    @staticmethod
    def check_null(X):
        
        if X.isnull().sum().sum() > 0:
            print(f"\nMissing values found: {X.isnull().sum().sum()}")
            X = X.fillna(X.mean())
        return X
        


class DataTransformations:
    """Utility class for data preprocessing and transformations for neural networks"""
    
    @staticmethod
    def standardize(X_train, X_val=None, X_test=None):
        """
        Standardize features to have mean=0 and std=1
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
        
        Returns:
            Tuple of (scaler, X_train_scaled, X_val_scaled, X_test_scaled)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        X_val_scaled = scaler.transform(X_val) if X_val is not None else None
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
        
        return scaler, X_train_scaled, X_val_scaled, X_test_scaled
    
    @staticmethod
    def apply_log_transform(X, features=None, offset=1.0):
        """
        Apply log1p transformation to handle heavy-tailed distributions
        
        Args:
            X: Feature matrix (pandas DataFrame or numpy array)
            features: List of feature names/indices to transform (None = all)
            offset: Offset to add before log (default=1.0 for log1p)
        
        Returns:
            Transformed feature matrix
        """
        X_transformed = X.copy()
        
        if isinstance(X, pd.DataFrame):
            if features is None:
                features = X.columns
            for feature in features:
                # Only apply to non-negative or shift to make non-negative
                min_val = X[feature].min()
                if min_val < 0:
                    X_transformed[feature] = np.log1p(X[feature] - min_val + offset)
                else:
                    X_transformed[feature] = np.log1p(X[feature])
        else:
            # Numpy array
            if features is None:
                features = range(X.shape[1])
            for idx in features:
                min_val = X[:, idx].min()
                if min_val < 0:
                    X_transformed[:, idx] = np.log1p(X[:, idx] - min_val + offset)
                else:
                    X_transformed[:, idx] = np.log1p(X[:, idx])
        
        return X_transformed
    
    @staticmethod
    def apply_quantile_transform(X_train, X_val=None, X_test=None, n_quantiles=1000, output_distribution='normal'):
        """
        Apply quantile transformation to map to uniform or normal distribution
        Useful for features with heavy tails or weird distributions
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            n_quantiles: Number of quantiles
            output_distribution: 'uniform' or 'normal'
        
        Returns:
            Tuple of (transformer, X_train_transformed, X_val_transformed, X_test_transformed)
        """
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            random_state=42
        )
        
        X_train_transformed = transformer.fit_transform(X_train)
        X_val_transformed = transformer.transform(X_val) if X_val is not None else None
        X_test_transformed = transformer.transform(X_test) if X_test is not None else None
        
        return transformer, X_train_transformed, X_val_transformed, X_test_transformed
    
    @staticmethod
    def apply_power_transform(X_train, X_val=None, X_test=None, method='yeo-johnson'):
        """
        Apply power transformation (Box-Cox or Yeo-Johnson) to make data more Gaussian-like
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            method: 'yeo-johnson' (works with negative values) or 'box-cox' (positive only)
        
        Returns:
            Tuple of (transformer, X_train_transformed, X_val_transformed, X_test_transformed)
        """
        transformer = PowerTransformer(method=method, standardize=True)
        
        X_train_transformed = transformer.fit_transform(X_train)
        X_val_transformed = transformer.transform(X_val) if X_val is not None else None
        X_test_transformed = transformer.transform(X_test) if X_test is not None else None
        
        return transformer, X_train_transformed, X_val_transformed, X_test_transformed
    
    @staticmethod
    def detect_skewed_features(X, threshold=1.0):
        """
        Detect features with high skewness that might benefit from transformation
        
        Args:
            X: Feature matrix (pandas DataFrame)
            threshold: Skewness threshold (default=1.0)
        
        Returns:
            Dictionary with feature names and their skewness values
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for skewness detection")
        
        from scipy.stats import skew
        
        skewness = {}
        for col in X.columns:
            skew_val = skew(X[col].dropna())
            if abs(skew_val) > threshold:
                skewness[col] = skew_val
        
        return dict(sorted(skewness.items(), key=lambda x: abs(x[1]), reverse=True))
    
    @staticmethod
    def create_validation_split(X_train, y_train, val_size=0.2, random_state=42):
        """
        Create a validation split from training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            val_size: Fraction of training data to use for validation
            random_state: Random seed
        
        Returns:
            Tuple of (X_train_new, X_val, y_train_new, y_val)
        """
        X_train_new, X_val, y_train_new, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size, 
            random_state=random_state,
            stratify=y_train
        )
        return X_train_new, X_val, y_train_new, y_val

