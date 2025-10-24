"""
Demo script to test the Neural Network implementation
Includes overfitting detection and cross-validation
"""

import sys
sys.path.append('src')

from utils import Utils, DataTransformations
from nn import NeuralNetClassifier, train_baseline_nn, kfold_cross_validation
import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("NEURAL NETWORK DEMO - HCP Belle Classification")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    data_path = 'data/data_hep - data_hep.csv'
    df = Utils.data_import(data_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Prepare binary classification
    print("\n2. Preparing binary classification...")
    X, y = Utils.bin_classification(df)
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Signal (0): {(y == 1).sum()}, Background (1): {(y == 0).sum()}")
    
    # Split data into train and test
    print("\n3. Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = Utils.data_split(X, y, ratio=0.3)
    print(f"   Train set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Create validation split from training data
    print("\n4. Creating validation split (20% of train)...")
    X_train_new, X_val, y_train_new, y_val = DataTransformations.create_validation_split(
        X_train, y_train, val_size=0.2
    )
    print(f"   Train set: {X_train_new.shape}")
    print(f"   Validation set: {X_val.shape}")
    
    # Convert to numpy arrays
    X_train_np = X_train_new.values
    X_val_np = X_val.values
    X_test_np = X_test.values
    y_train_np = y_train_new.values
    y_val_np = y_val.values
    y_test_np = y_test.values
    
    # Train the baseline neural network
    print("\n5. Training Baseline Neural Network...")
    print("-" * 80)
    
    nn_model, test_metrics, overfitting_analysis = train_baseline_nn(
        X_train=X_train_np,
        y_train=y_train_np,
        X_val=X_val_np,
        y_val=y_val_np,
        X_test=X_test_np,
        y_test=y_test_np,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        epochs=50,
        early_stopping_patience=5,
        save_model_path='models/baseline_nn.pt',
        plot_dir='plots'
    )
    
    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETED!")
    print("="*80)
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    
    # Perform k-fold cross-validation for stability assessment
    print("\n\n" + "="*80)
    print("PERFORMING K-FOLD CROSS-VALIDATION")
    print("="*80)
    print("This helps assess model stability and detect overfitting...")
    
    # Use smaller subset for faster demo (optional - remove for full validation)
    # Uncomment the next 2 lines to use full dataset
    # X_full = np.vstack([X_train_np, X_val_np])
    # y_full = np.hstack([y_train_np, y_val_np])
    
    # For demo, use a sample
    sample_size = min(100000, len(X_train_np))
    indices = np.random.choice(len(X_train_np), sample_size, replace=False)
    X_sample = X_train_np[indices]
    y_sample = y_train_np[indices]
    
    cv_results = kfold_cross_validation(
        X=X_sample,
        y=y_sample,
        n_splits=5,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        epochs=30,  # Fewer epochs for CV
        early_stopping_patience=5
    )
    
    print("\n\n" + "="*80)
    print("ALL ANALYSES COMPLETED!")
    print("="*80)
    print("\n📊 Results Summary:")
    print(f"  • Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  • Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  • CV Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"  • CV Mean ROC AUC: {cv_results['mean_roc_auc']:.4f} ± {cv_results['std_roc_auc']:.4f}")
    
    print("\n📁 Output Files:")
    print("  • Model: models/baseline_nn.pt")
    print("  • Plots: plots/")
    print("    - nn_training_history.png (learning curves with overfitting indicators)")
    print("    - confusion_matrix_*.png")
    print("    - roc_curve_*.png")
    print("    - signal_vs_background_*.png")
    
    print("\n✅ Training pipeline completed successfully!")
    

if __name__ == "__main__":
    main()
