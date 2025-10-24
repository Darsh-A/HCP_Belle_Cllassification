# train_classifier.py

import pandas as pd
import numpy as np
from PyFastBDT import FastBDT
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import os

from .utils import Utils

@staticmethod
def FastBDT_binary_classification(df,plotname="fastbdt_roc_curve.png"):
    print("--- FastBDT Binary Classification ---")

    X, y = Utils.bin_classification(df)

    print("\nData distribution:")
    print(y.value_counts())
    print(f"\nUsing {len(X.columns)} features for training.")

    X_train, X_test, y_train, y_test = Utils.data_split(X, y, ratio=0.3)

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

    Utils.plot_roc(y_test, y_pred_scores,plotname=plotname)

    print("\nIntern Feature Importance:")
    print(bdt.internFeatureImportance()) 
# FastBDT_binary_classification(
#     df = Utils.data_import('../data/data_hep - data_hep.csv')
# )