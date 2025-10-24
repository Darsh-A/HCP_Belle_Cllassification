"""
Data Cleaning and Preprocessing Class
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessing:
    """
    A class for data analysis, cleaning, and preprocessing operations.
    All methods are static, can use them without instantiating the class.
    """
    
    @staticmethod
    def data_correlation(df, threshold=0.9, preview=False):
        """Find highly correlated feature pairs above the threshold"""
        features = df.drop(columns=['index', 'type'])
        corr_matrix = features.corr()
        corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1]
        high_corr = corr_pairs[abs(corr_pairs) > threshold]
        if preview == True:
            plt.figure(figsize=(10, 10))
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.show()
        return high_corr
    
    @staticmethod
    def data_reduction_correlation(df, threshold=0.9, preview=False):
        """Reduce features by removing highly correlated ones"""
        features = df.drop(columns=['index', 'type'])
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        reduced_features = features.drop(columns=to_drop)
        if preview == True:
            plt.figure(figsize=(10, 10))
            sns.heatmap(reduced_features.corr(), cmap='coolwarm', center=0)
            plt.title('Reduced Correlation Matrix')
            plt.show()
        return reduced_features

class DataCheck:

    @staticmethod
    def data_check(df):
        """Check the data for missing values, duplicates, and outliers"""
        print(f"Missing values: {df.isnull().sum()}")
        print(f"Duplicates: {df.duplicated().sum()}")

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Column {col} is not numeric")
                print(df[col].value_counts())
        
        return None

class DataAnalysis:

    @staticmethod
    def binary_classf_preview(df):
        types = df['type'].value_counts()
        """
        BINARY CLASSIFICATION
        0,1 = signal
        2,3,4,5 = background
        """
        print("Signal = ", types[0]+types[1])
        print("Background = ", types[2]+types[3]+types[4]+types[5])
        return types


class DataVisualization:

    def feature_distribution_by_type(df, full_df=False):
        signal_colors = ["#d62728", "#ff9896"]
        background_colors = sns.color_palette("Purples", n_colors=4)  # 4 shades of purple

        custom_palette = {
            0: signal_colors[0],   # type 0
            1: signal_colors[1],   # type 1
            2: background_colors[0],
            3: background_colors[1],
            4: background_colors[2],
            5: background_colors[3],
        }
        if full_df == True:
            features = df.drop(columns=['index', 'type'])
        else:
            features = DataPreprocessing.data_reduction_correlation(df, threshold=0.9)
        num_features = features.shape[1]
        num_cols = 3
        num_rows = (num_features + num_cols - 1) // num_cols

        print("num_features: ", num_features, "num_cols: ", num_cols, "num_rows: ", num_rows)

        plt.figure(figsize=(20, num_rows * 5))
        for i, column in enumerate(features.columns):
            plt.subplot(num_rows, num_cols, i + 1)
            sns.histplot(
                data=df,
                x=column,
                hue='type',
                element='step',
                stat='density',
                common_norm=False,
                palette=custom_palette
            )
            plt.title(f'Distribution of {column} by Type')

        plt.tight_layout()
        # plt.savefig("feature_distributions_by_type.png")
        plt.show()

