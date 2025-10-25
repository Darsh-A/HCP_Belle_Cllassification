"""
Neural Network Models for Binary Classification
Implements MLP baseline with PyTorch for tabular data classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    accuracy_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Optional, List
import copy


class TabularDataset(Dataset):
    """
    Custom PyTorch Dataset for tabular data
    Converts numpy arrays to PyTorch tensors
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Feature matrix (numpy array)
            y: Target labels (numpy array)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BaselineMLP(nn.Module):
    """
    Baseline Multi-Layer Perceptron for binary classification
    Architecture: Input -> 128 -> 64 -> 32 -> Output
    With ReLU activations and Dropout for regularization
    """
    def __init__(
        self, 
        input_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        output_size: int = 1
    ):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout probability (0.0-1.0)
            output_size: Number of output neurons (1 for binary with BCEWithLogitsLoss)
        """
        super(BaselineMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (no activation - using BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Current model
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss improved ({self.best_loss:.6f} -> {val_loss:.6f})')
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        
        return self.early_stop


class NeuralNetClassifier:
    """
    High-level wrapper for neural network binary classification
    Provides sklearn-like interface for easy integration
    """
    def __init__(
        self,
        hidden_sizes: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        device: str = None,
        random_state: int = 42
    ):
        """
        Args:
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout probability
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization strength
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
            device: 'cuda', 'cpu', or None (auto-detect)
            random_state: Random seed for reproducibility
        """
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        # Initialize placeholders
        self.model = None
        self.scaler = StandardScaler()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_auc': [],
            'val_auc': []
        }
    
    def _create_data_loaders(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create PyTorch DataLoaders for training and validation"""
        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TabularDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        return train_loader, val_loader
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
        
        Returns:
            self
        """
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = BaselineMLP(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            verbose=verbose
        )
        
        # Training loop
        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Model architecture: {input_size} -> {' -> '.join(map(str, self.hidden_sizes))} -> 1")
            print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
            print("\nStarting training...")
            print("-" * 80)
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            train_probs = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * batch_X.size(0)
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                train_probs.extend(probs.flatten())
                train_preds.extend(preds.flatten())
                train_targets.extend(batch_y.cpu().numpy().flatten())
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = accuracy_score(train_targets, train_preds)
            train_auc = roc_auc_score(train_targets, train_probs)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_auc'].append(train_auc)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []
                val_probs = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device).unsqueeze(1)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item() * batch_X.size(0)
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        preds = (probs > 0.5).astype(int)
                        
                        val_probs.extend(probs.flatten())
                        val_preds.extend(preds.flatten())
                        val_targets.extend(batch_y.cpu().numpy().flatten())
                
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = accuracy_score(val_targets, val_preds)
                val_auc = roc_auc_score(val_targets, val_probs)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_auc'].append(val_auc)
                
                # Print progress
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}]")
                    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
                    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val AUC:   {val_auc:.4f}")
                
                # Early stopping check
                if early_stopping(val_loss, self.model):
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            else:
                # No validation set
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}]")
                    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
        
        # Restore best model if early stopping was used
        if val_loader is not None and early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            if verbose:
                print(f"\nRestored best model with validation loss: {early_stopping.best_loss:.4f}")
        
        if verbose:
            print("-" * 80)
            print("Training completed!")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        dataset = TabularDataset(X_scaled, np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs.flatten())
        
        return np.array(all_probs)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
        
        Returns:
            Array of predicted class labels
        """
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)
    
    def check_overfitting(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        """
        Check for overfitting by comparing training and validation/test performance
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print detailed analysis
        
        Returns:
            Dictionary with overfitting metrics
        """
        if not self.history['train_loss'] or not self.history['val_loss']:
            print("Warning: No training history available for overfitting analysis")
            return None
        
        # Get training set predictions
        y_train_proba = self.predict_proba(X_train)
        y_train_pred = (y_train_proba > 0.5).astype(int)
        
        train_acc_final = accuracy_score(y_train, y_train_pred)
        train_auc_final = roc_auc_score(y_train, y_train_proba)
        
        # Get final validation metrics
        val_acc_final = self.history['val_acc'][-1] if self.history['val_acc'] else None
        val_auc_final = self.history['val_auc'][-1] if self.history['val_auc'] else None
        train_loss_final = self.history['train_loss'][-1]
        val_loss_final = self.history['val_loss'][-1] if self.history['val_loss'] else None
        
        # Calculate gaps
        acc_gap = train_acc_final - val_acc_final if val_acc_final else None
        auc_gap = train_auc_final - val_auc_final if val_auc_final else None
        loss_gap = val_loss_final - train_loss_final if val_loss_final else None
        
        # Determine overfitting severity
        overfitting_score = 0
        warnings = []
        
        if auc_gap and auc_gap > 0.05:
            overfitting_score += 1
            warnings.append(f"Large AUC gap: {auc_gap:.4f}")
        if auc_gap and auc_gap > 0.10:
            overfitting_score += 1
            warnings.append(f"Very large AUC gap: {auc_gap:.4f}")
        
        if loss_gap and loss_gap > 0.1:
            overfitting_score += 1
            warnings.append(f"Validation loss higher than training: {loss_gap:.4f}")
        
        # Check if validation loss increased over last epochs
        if len(self.history['val_loss']) > 5:
            recent_val_loss = self.history['val_loss'][-5:]
            if recent_val_loss[-1] > recent_val_loss[0]:
                overfitting_score += 1
                warnings.append("Validation loss increased in recent epochs")
        
        if verbose:
            print("\n" + "="*80)
            print("OVERFITTING ANALYSIS")
            print("="*80)
            print(f"\nTraining Set Performance:")
            print(f"  Accuracy: {train_acc_final:.4f}")
            print(f"  ROC AUC:  {train_auc_final:.4f}")
            print(f"  Loss:     {train_loss_final:.4f}")
            
            if val_acc_final:
                print(f"\nValidation Set Performance:")
                print(f"  Accuracy: {val_acc_final:.4f}")
                print(f"  ROC AUC:  {val_auc_final:.4f}")
                print(f"  Loss:     {val_loss_final:.4f}")
                
                print(f"\nPerformance Gaps (Train - Val):")
                print(f"  Accuracy Gap: {acc_gap:+.4f}")
                print(f"  AUC Gap:      {auc_gap:+.4f}")
                print(f"  Loss Gap:     {loss_gap:+.4f}")
            
            print(f"\nOverfitting Score: {overfitting_score}/4")
            if overfitting_score == 0:
                print("✓ No significant overfitting detected")
            elif overfitting_score == 1:
                print("⚠ Mild overfitting detected")
            elif overfitting_score == 2:
                print("⚠⚠ Moderate overfitting detected")
            else:
                print("⚠⚠⚠ Severe overfitting detected")
            
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  • {warning}")
            
            print("\nRecommendations:")
            if overfitting_score >= 2:
                print("  • Increase dropout rate (currently: {:.2f})".format(self.dropout_rate))
                print("  • Increase weight decay / L2 regularization")
                print("  • Reduce model complexity (fewer/smaller layers)")
                print("  • Use more training data or data augmentation")
                print("  • Apply stronger early stopping")
            elif overfitting_score == 1:
                print("  • Consider slight increase in dropout or weight decay")
                print("  • Monitor validation metrics closely")
            else:
                print("  • Current regularization appears adequate")
                print("  • Could potentially increase model capacity if needed")
        
        return {
            'train_accuracy': train_acc_final,
            'train_auc': train_auc_final,
            'train_loss': train_loss_final,
            'val_accuracy': val_acc_final,
            'val_auc': val_auc_final,
            'val_loss': val_loss_final,
            'accuracy_gap': acc_gap,
            'auc_gap': auc_gap,
            'loss_gap': loss_gap,
            'overfitting_score': overfitting_score,
            'warnings': warnings
        }
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        plot_dir: str = 'plots',
        model_name: str = 'NN'
    ):
        """
        Evaluate model on test set and generate plots
        
        Args:
            X_test: Test features
            y_test: Test labels
            plot_dir: Directory to save plots
            model_name: Name for plot files
        
        Returns:
            Dictionary with evaluation metrics
        """
        os.makedirs(plot_dir, exist_ok=True)
        
        # Get predictions
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n" + "="*80)
        print(f"{model_name} MODEL EVALUATION")
        print("="*80)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Signal (B mesons)', 'Background (continuum)']
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(plot_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(cm_path)
        print(f"\nConfusion matrix saved to '{cm_path}'")
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Background Efficiency)')
        plt.ylabel('True Positive Rate (Signal Efficiency)')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        roc_path = os.path.join(plot_dir, f'roc_curve_{model_name}.png')
        plt.savefig(roc_path)
        print(f"ROC curve saved to '{roc_path}'")
        plt.close()
        
        # Signal Efficiency vs Background Rejection
        signal_eff = tpr
        background_rej = 1 - fpr
        
        plt.figure(figsize=(10, 6))
        plt.plot(signal_eff, background_rej, lw=2, color='darkgreen')
        plt.xlabel('Signal Efficiency')
        plt.ylabel('Background Rejection (1 - FPR)')
        plt.title(f'{model_name} Signal Efficiency vs Background Rejection')
        plt.grid(alpha=0.3)
        
        # Add some key points
        for sig_eff_target in [0.5, 0.7, 0.9]:
            idx = np.argmin(np.abs(signal_eff - sig_eff_target))
            bkg_rej = background_rej[idx]
            plt.plot(signal_eff[idx], bkg_rej, 'ro', markersize=8)
            plt.annotate(
                f'Sig Eff={signal_eff[idx]:.2f}\nBkg Rej={bkg_rej:.2f}',
                xy=(signal_eff[idx], bkg_rej),
                xytext=(10, -10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )
        
        sig_bkg_path = os.path.join(plot_dir, f'signal_vs_background_{model_name}.png')
        plt.savefig(sig_bkg_path)
        print(f"Signal efficiency plot saved to '{sig_bkg_path}'")
        plt.close()
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=['Signal', 'Background'],
                output_dict=True
            )
        }
    
    def plot_training_history(self, save_path: str = 'plots/training_history.png'):
        """
        Plot training history (loss, accuracy, AUC over epochs)
        Enhanced with overfitting indicators
        
        Args:
            save_path: Path to save the plot
        """
        if not self.history['train_loss']:
            print("No training history available. Train the model first.")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        has_val = bool(self.history['val_loss'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if has_val:
            axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            
            # Highlight overfitting region (where val loss > train loss)
            train_loss = np.array(self.history['train_loss'])
            val_loss = np.array(self.history['val_loss'])
            overfit_mask = val_loss > train_loss
            if np.any(overfit_mask):
                axes[0].fill_between(epochs, train_loss, val_loss, 
                                    where=overfit_mask, alpha=0.3, color='red',
                                    label='Overfitting Region')
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        if has_val:
            axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            
            # Show gap
            final_gap = self.history['train_acc'][-1] - self.history['val_acc'][-1]
            axes[1].text(0.05, 0.05, f'Final Gap: {final_gap:.4f}', 
                        transform=axes[1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # AUC
        axes[2].plot(epochs, self.history['train_auc'], 'b-', label='Training AUC', linewidth=2)
        if has_val:
            axes[2].plot(epochs, self.history['val_auc'], 'r-', label='Validation AUC', linewidth=2)
            
            # Show gap
            final_gap = self.history['train_auc'][-1] - self.history['val_auc'][-1]
            axes[2].text(0.05, 0.05, f'Final Gap: {final_gap:.4f}', 
                        transform=axes[2].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('ROC AUC', fontsize=12)
        axes[2].set_title('Training and Validation ROC AUC', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to '{save_path}'")
        plt.close()
    
    def save_model(self, path: str = 'models/nn_model.pt'):
        """
        Save model weights and scaler
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'history': self.history
        }, path)
        print(f"Model saved to '{path}'")
    
    def load_model(self, path: str, input_size: int):
        """
        Load model weights and scaler
        
        Args:
            path: Path to load the model from
            input_size: Number of input features
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.hidden_sizes = checkpoint['hidden_sizes']
        self.dropout_rate = checkpoint['dropout_rate']
        self.scaler = checkpoint['scaler']
        self.history = checkpoint['history']
        
        self.model = BaselineMLP(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from '{path}'")


# Convenience function for quick training
def train_baseline_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_sizes: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 50,
    early_stopping_patience: int = 5,
    save_model_path: str = 'models/baseline_nn.pt',
    plot_dir: str = 'plots'
) -> Tuple[NeuralNetClassifier, dict]:
    """
    Convenience function to train and evaluate a baseline neural network
    
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    print("="*80)
    print("BASELINE NEURAL NETWORK TRAINING")
    print("="*80)
    
    # Initialize classifier
    nn_clf = NeuralNetClassifier(
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience
    )
    
    # Train
    nn_clf.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    # Plot training history
    nn_clf.plot_training_history(save_path=os.path.join(plot_dir, 'nn_training_history.png'))
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    val_metrics = nn_clf.evaluate(X_val, y_val, plot_dir=plot_dir, model_name='NN_Validation')
    
    # Check for overfitting
    overfitting_analysis = nn_clf.check_overfitting(X_train, y_train, verbose=True)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    test_metrics = nn_clf.evaluate(X_test, y_test, plot_dir=plot_dir, model_name='NN_Test')
    
    # Save model
    nn_clf.save_model(save_model_path)
    
    return nn_clf, test_metrics, overfitting_analysis


def train_neural_network(
    data,
    hidden_sizes: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 50,
    early_stopping_patience: int = 5,
    importance_threshold: float = 0.01,
    random_state: int = 42
):
    """
    Train a Neural Network model on the provided data.
    
    Args:
        data: pandas DataFrame containing the dataset
        hidden_sizes: List of hidden layer sizes
        dropout_rate: Dropout probability
        learning_rate: Learning rate for Adam optimizer
        weight_decay: L2 regularization strength
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        early_stopping_patience: Patience for early stopping
        importance_threshold: Threshold for feature importance (not applicable for NN, included for API consistency)
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with keys 'confusion_matrix', 'roc_auc_score', 'accuracy', 'model',
              'feature_importance', 'reduced_features'
    """
    from utils import Utils, DataTransformations
    
    # Prepare data
    X, y = Utils.bin_classification(data)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = Utils.data_split(X, y, ratio=0.3)
    
    # Create validation split from training data
    X_train_new, X_val, y_train_new, y_val = DataTransformations.create_validation_split(
        X_train, y_train, val_size=0.2, random_state=random_state
    )
    
    # Convert to numpy arrays
    X_train_np = X_train_new.values
    X_val_np = X_val.values
    X_test_np = X_test.values
    y_train_np = y_train_new.values
    y_val_np = y_val.values
    y_test_np = y_test.values
    
    # Initialize classifier
    nn_clf = NeuralNetClassifier(
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        random_state=random_state
    )
    
    # Train
    nn_clf.fit(X_train_np, y_train_np, X_val_np, y_val_np, verbose=False)
    
    # Get predictions on test set
    y_pred_proba = nn_clf.predict_proba(X_test_np)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    cm = Utils.give_conf_matrix(y_test_np, y_pred)
    roc_auc_score = Utils.give_roc_auc_score(y_test_np, y_pred_proba)
    accuracy = Utils.give_accuracy(y_test_np, y_pred)
    
    # Neural networks don't have traditional feature importance like tree models
    # We can calculate gradient-based importance or use input perturbation
    # For simplicity and API consistency, return empty feature importance
    feature_names = X.columns.tolist()
    feature_importance = {feature: 0.0 for feature in feature_names}
    reduced_features = []  # NN doesn't support feature reduction in the same way
    
    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "neural_network_model.pt")
    nn_clf.save_model(model_path)
    
    # Return results as dictionary
    return {
        'confusion_matrix': cm,
        'roc_auc_score': roc_auc_score,
        'accuracy': accuracy,
        'model': nn_clf,
        'feature_importance': feature_importance,
        'reduced_features': reduced_features
    }


def kfold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    hidden_sizes: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 50,
    early_stopping_patience: int = 5,
    random_state: int = 42
) -> dict:
    """
    Perform k-fold cross-validation to assess model stability and generalization
    
    Args:
        X: Feature matrix
        y: Target labels
        n_splits: Number of folds
        ... (other hyperparameters same as NeuralNetClassifier)
    
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("="*80)
    print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
    print("="*80)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = {
        'accuracy': [],
        'roc_auc': [],
        'train_accuracy': [],
        'train_auc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*80}")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Train model
        nn_clf = NeuralNetClassifier(
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state + fold  # Different seed per fold
        )
        
        nn_clf.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, verbose=False)
        
        # Evaluate on validation fold
        y_val_proba = nn_clf.predict_proba(X_val_fold)
        y_val_pred = (y_val_proba > 0.5).astype(int)
        
        val_acc = accuracy_score(y_val_fold, y_val_pred)
        val_auc = roc_auc_score(y_val_fold, y_val_proba)
        
        # Also get training performance
        y_train_proba = nn_clf.predict_proba(X_train_fold)
        y_train_pred = (y_train_proba > 0.5).astype(int)
        
        train_acc = accuracy_score(y_train_fold, y_train_pred)
        train_auc = roc_auc_score(y_train_fold, y_train_proba)
        
        fold_results['accuracy'].append(val_acc)
        fold_results['roc_auc'].append(val_auc)
        fold_results['train_accuracy'].append(train_acc)
        fold_results['train_auc'].append(train_auc)
        
        print(f"\nFold {fold} Results:")
        print(f"  Train Accuracy: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
        print(f"  Val Accuracy:   {val_acc:.4f} | Val AUC:   {val_auc:.4f}")
        print(f"  Accuracy Gap:   {train_acc - val_acc:+.4f}")
        print(f"  AUC Gap:        {train_auc - val_auc:+.4f}")
    
    # Calculate statistics
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    print("\nValidation Performance Across Folds:")
    print(f"  Accuracy: {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")
    print(f"  ROC AUC:  {np.mean(fold_results['roc_auc']):.4f} ± {np.std(fold_results['roc_auc']):.4f}")
    
    print("\nTraining Performance Across Folds:")
    print(f"  Accuracy: {np.mean(fold_results['train_accuracy']):.4f} ± {np.std(fold_results['train_accuracy']):.4f}")
    print(f"  ROC AUC:  {np.mean(fold_results['train_auc']):.4f} ± {np.std(fold_results['train_auc']):.4f}")
    
    avg_acc_gap = np.mean(np.array(fold_results['train_accuracy']) - np.array(fold_results['accuracy']))
    avg_auc_gap = np.mean(np.array(fold_results['train_auc']) - np.array(fold_results['roc_auc']))
    
    print("\nAverage Performance Gaps (Train - Val):")
    print(f"  Accuracy Gap: {avg_acc_gap:+.4f}")
    print(f"  AUC Gap:      {avg_auc_gap:+.4f}")
    
    # Stability check
    acc_std = np.std(fold_results['accuracy'])
    auc_std = np.std(fold_results['roc_auc'])
    
    print("\nModel Stability Assessment:")
    if acc_std < 0.01 and auc_std < 0.01:
        print("  ✓ Excellent stability across folds")
    elif acc_std < 0.02 and auc_std < 0.02:
        print("  ✓ Good stability across folds")
    elif acc_std < 0.05 and auc_std < 0.05:
        print("  ⚠ Moderate stability - some variance across folds")
    else:
        print("  ⚠⚠ Poor stability - high variance across folds")
        print("     Consider: more data, simpler model, or different preprocessing")
    
    # Overfitting check
    if avg_auc_gap > 0.1:
        print("\nOverfitting Warning:")
        print("  ⚠⚠ Large train-val gap suggests overfitting")
        print("     Consider: stronger regularization, more dropout, or less model complexity")
    elif avg_auc_gap > 0.05:
        print("\nOverfitting Warning:")
        print("  ⚠ Moderate train-val gap - monitor closely")
    else:
        print("\nOverfitting Check:")
        print("  ✓ Train-val gap is acceptable")
    
    return {
        'fold_results': fold_results,
        'mean_accuracy': np.mean(fold_results['accuracy']),
        'std_accuracy': np.std(fold_results['accuracy']),
        'mean_roc_auc': np.mean(fold_results['roc_auc']),
        'std_roc_auc': np.std(fold_results['roc_auc']),
        'mean_train_accuracy': np.mean(fold_results['train_accuracy']),
        'mean_train_auc': np.mean(fold_results['train_auc']),
        'avg_accuracy_gap': avg_acc_gap,
        'avg_auc_gap': avg_auc_gap
    }


if __name__ == "__main__":
    # Example usage
    from utils import Utils
    
    df = Utils.data_import('data/data_hep - data_hep.csv')
    
    # Train Neural Network
    nn_results = train_neural_network(
        df,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=1e-3,
        epochs=50
    )
    
    print("\nNeural Network Results:")
    print(f"Accuracy: {nn_results['accuracy']:.4f}")
    print(f"ROC-AUC: {nn_results['roc_auc_score']:.4f}")
    print(f"Confusion Matrix:\n{nn_results['confusion_matrix']}")
