"""
Any Misc function goes here
like decorations, utilities, etc.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

class Utils:

    @staticmethod
    def data_import(data):
        """Import CSV data into a pandas DataFrame"""
        df = pd.read_csv(data)
        return df
    
    @staticmethod
    def data_split(x,y,ratio=0.3):
        X_train, X_test, y_train, y_test = train_test_split(
             x, y, test_size=ratio, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def bin_classification(df):
        """Binary classification: 0,1 = signal; 2,3,4,5 = background"""
        df['is_signal'] = df['type'].isin([0, 1]).astype(int)
        y = df['is_signal']
        features_to_drop = ['index', 'type', 'is_signal']
        X = df.drop(columns=features_to_drop)
        return X, y
    
    def plot_roc(y_test, y_pred_scores):
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
        plt.savefig("graphs/fastbdt_roc_curve.png")
        print("Plot saved as 'fastbdt_roc_curve.png'")
        plt.show()

        return roc_auc
