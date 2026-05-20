import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA

from preprocessing import run_preprocessing

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ─────────────────────────────────────────────
# MODEL 1: Decision Tree Classifier
# ─────────────────────────────────────────────

def train_decision_tree(X_train, X_test, yc_train, yc_test, feature_names, encoder):
    print("\n" + "=" * 50)
    print("MODEL 1: DECISION TREE CLASSIFIER")
    print("=" * 50)

    clf = DecisionTreeClassifier(max_depth=15, random_state=42)
    clf.fit(X_train, yc_train)

    y_pred = clf.predict(X_test)

    accuracy  = accuracy_score(yc_test, y_pred)
    precision = precision_score(yc_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(yc_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"\nClassification Report:\n{classification_report(yc_test, y_pred, target_names=encoder.classes_, zero_division=0)}")

    # Feature importance bar chart
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(feature_names)), importances[indices], color='steelblue', edgecolor='white')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=30, ha='right')
    ax.set_title('Decision Tree — Feature Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance Score')
    ax.set_ylim(0, importances.max() * 1.15)
    for i, v in enumerate(importances[indices]):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
    plt.close()
    print("[Decision Tree] feature_importance.png saved.")

    # Serialize
    joblib.dump(clf, os.path.join(MODELS_DIR, 'decision_tree.pkl'))
    print("[Decision Tree] Model saved to models/decision_tree.pkl")

    return clf, {'accuracy': accuracy, 'precision': precision, 'recall': recall}


# ─────────────────────────────────────────────
# MODEL 2: KMeans Clustering
# ─────────────────────────────────────────────

def find_optimal_k(X_all, k_range=range(2, 10)):
    """Elbow method to find the best number of clusters."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_all)
        inertias.append(km.inertia_)
    return inertias


def train_kmeans(X_all, n_clusters=4):
    print("\n" + "=" * 50)
    print("MODEL 2: KMEANS CLUSTERING")
    print("=" * 50)

    # Elbow plot
    k_range = range(2, 10)
    inertias = find_optimal_k(X_all, k_range)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker='o', color='darkorange', linewidth=2)
    ax.axvline(x=n_clusters, color='red', linestyle='--', label=f'Chosen k={n_clusters}')
    ax.set_title('KMeans — Elbow Method', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'elbow_plot.png'), dpi=150)
    plt.close()

    # Train final model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_all)

    sil_score = silhouette_score(X_all, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Cluster sizes: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")

    # Cluster scatter plot via PCA (2D)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_all)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    zone_labels = ['Zone A — Fertile', 'Zone B — Moderate', 'Zone C — Dry', 'Zone D — Saline']

    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[i], label=zone_labels[i], alpha=0.6, s=20, edgecolors='none')
    centers_2d = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='black', marker='X', s=180, zorder=5, label='Centroids')
    ax.set_title(f'Soil Profile Clusters (k={n_clusters}, Silhouette={sil_score:.3f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cluster_scatter.png'), dpi=150)
    plt.close()
    print("[KMeans] cluster_scatter.png saved.")

    # Serialize
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans.pkl'))
    joblib.dump(pca,    os.path.join(MODELS_DIR, 'pca.pkl'))
    print("[KMeans] Model saved to models/kmeans.pkl")

    return kmeans, {'silhouette_score': sil_score, 'n_clusters': n_clusters}


# ─────────────────────────────────────────────
# MODEL 3: Linear Regression (Yield Prediction)
# ─────────────────────────────────────────────

def train_linear_regression(X_train, X_test, yy_train, yy_test):
    print("\n" + "=" * 50)
    print("MODEL 3: LINEAR REGRESSION (YIELD PREDICTION)")
    print("=" * 50)

    reg = LinearRegression()
    reg.fit(X_train, yy_train)

    y_pred = reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(yy_test, y_pred))
    mae  = mean_absolute_error(yy_test, y_pred)
    r2   = r2_score(yy_test, y_pred)

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # Residual plot
    residuals = yy_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Actual vs Predicted
    axes[0].scatter(yy_test, y_pred, alpha=0.4, color='steelblue', s=15, edgecolors='none')
    lims = [min(yy_test.min(), y_pred.min()), max(yy_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    axes[0].set_title('Actual vs Predicted Yield', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Actual Yield (kg/ha)')
    axes[0].set_ylabel('Predicted Yield (kg/ha)')
    axes[0].legend()

    # Right: Residuals vs Fitted
    axes[1].scatter(y_pred, residuals, alpha=0.4, color='darkorange', s=15, edgecolors='none')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residuals vs Fitted Values', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Fitted Values (Predicted Yield)')
    axes[1].set_ylabel('Residuals')

    fig.suptitle(f'Linear Regression — RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residual_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Linear Regression] residual_plot.png saved.")

    # Serialize
    joblib.dump(reg, os.path.join(MODELS_DIR, 'linear_regression.pkl'))
    print("[Linear Regression] Model saved to models/linear_regression.pkl")

    return reg, {'rmse': rmse, 'mae': mae, 'r2': r2}


# ─────────────────────────────────────────────
# MAIN — Train all models
# ─────────────────────────────────────────────

def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = run_preprocessing()

    clf, dt_metrics  = train_decision_tree(
        data['X_train'], data['X_test'],
        data['yc_train'], data['yc_test'],
        data['feature_names'], data['encoder']
    )

    kmeans, km_metrics = train_kmeans(data['X_all'], n_clusters=4)

    reg, lr_metrics = train_linear_regression(
        data['X_train'], data['X_test'],
        data['yy_train'], data['yy_test']
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE — PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Decision Tree  -> Accuracy: {dt_metrics['accuracy']:.4f} | Precision: {dt_metrics['precision']:.4f} | Recall: {dt_metrics['recall']:.4f}")
    print(f"KMeans         -> Silhouette Score: {km_metrics['silhouette_score']:.4f} | Clusters: {km_metrics['n_clusters']}")
    print(f"Linear Reg.    -> RMSE: {lr_metrics['rmse']:.2f} | MAE: {lr_metrics['mae']:.2f} | R2: {lr_metrics['r2']:.4f}")
    print("\nAll models saved to models/")
    print("All plots saved to results/")

    return clf, kmeans, reg


if __name__ == '__main__':
    train_all()
