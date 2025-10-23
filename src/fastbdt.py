# train_classifier.py

import pandas as pd
import numpy as np
from PyFastBDT import FastBDT
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

from data_anan import DataPreprocessing

print("--- FastBDT Binary Classification ---")

df = DataPreprocessing.data_import('../data/data_hep - data_hep.csv') 


df['is_signal'] = df['type'].isin([0, 1]).astype(int)

print("\nData distribution:")
print(df['is_signal'].value_counts())


y = df['is_signal']

features_to_drop = ['index', 'type', 'is_signal']
X = df.drop(columns=features_to_drop)

print(f"\nUsing {len(X.columns)} features for training.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size:  {len(X_test)}")

bdt = FastBDT.Classifier()


print("\nTraining the FastBDT model...")
bdt.fit(X_train.values, y_train.values)
print("Training complete!")


y_pred_scores = bdt.predict(X_test.values)

# To calculate accuracy, we need to convert scores into class labels (0 or 1).
# A common threshold is 0.5.
y_pred_class = (y_pred_scores > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_class)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")


fpr, tpr, thresholds = roc_curve(y_test, y_pred_scores)
roc_auc = auc(fpr, tpr)

print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")

# Create the plot
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
plt.show() # Uncomment if you are in a graphical environment and want to see the plot immediately

print("\n--- Script finished ---")