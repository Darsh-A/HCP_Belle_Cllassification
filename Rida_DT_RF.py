import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 120

# ---------- Load Data ----------
def load_data(path='DataScience_Project_Data.csv'):
    """Load and return the dataset"""
    df = pd.read_csv(path)
    return df

def reduce_dimensionality(X, n_components=10):
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")
    return pd.DataFrame(X_reduced)

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', cbar=False, square=True)
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_prob, model_name):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

# ---------- Step 1: Load and prepare data ----------
df = load_data()

# Example: convert 'type' to binary class (modify based on your dataset)
df['binary_type'] = df['type'].apply(lambda x: 1 if x in [0, 1] else 0)

X = df.drop(columns=['type', 'binary_type'])
y = df['binary_type']

# ---------- Step 2: Split train/test ----------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------- Step 3: Dimensionality Reduction ----------
X_reduced = reduce_dimensionality(X)
xr_train, xr_test, yr_train, yr_test = train_test_split(
    X_reduced, y, test_size=0.3, random_state=42, stratify=y
)

# ---------- Step 4: Train Models ----------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

results = []

for name, model in models.items():
    # Without Dim Reduction
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    results.append({
        'Model': name,
        'Type': 'Original Features',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cm, f"{name} (Original Features)")

    plot_roc_curve(y_test, y_prob, f"{name} (Original)")

    # With Dim Reduction
    model.fit(xr_train, yr_train)
    y_pred_red = model.predict(xr_test)
    y_prob_red = model.predict_proba(xr_test)[:, 1]

    results.append({
        'Model': name,
        'Type': 'Reduced Features',
        'Accuracy': accuracy_score(yr_test, y_pred_red),
        'Precision': precision_score(yr_test, y_pred_red),
        'Recall': recall_score(yr_test, y_pred_red),
        'F1 Score': f1_score(yr_test, y_pred_red),
        'ROC-AUC': roc_auc_score(yr_test, y_prob_red)
    })


    cm = confusion_matrix(yr_test, y_pred_red)

    plot_confusion_matrix(cm, f"{name} (Reduced Features)")

    plot_roc_curve(yr_test, y_prob_red, f"{name} (Reduced)")

plt.tight_layout()
plt.show()


# ---------- Step 5: Feature Importance Plot ----------
rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)
importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.head(15), y=importances.head(15).index)
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ---------- Step 6: Final Summary Table ----------
# ---------- Step 6: Final Summary Table (PRETTY VERSION) ----------
results_df = pd.DataFrame(results)

# Identify only numeric columns
numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

styled = (
    results_df.style
    .background_gradient(subset=numeric_cols, cmap='YlGnBu')  # color gradient for performance metrics
    .format({col: "{:.3f}" for col in numeric_cols})           # safely format only numeric cols
    .set_table_styles([
        {'selector': 'th', 'props': [('font-size', '13px'), ('text-align', 'center'), ('background-color', '#f4f4f4')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '12px')]}
    ])
    .hide(axis="index")  # optional: hides the row index for a cleaner look
)

styled


