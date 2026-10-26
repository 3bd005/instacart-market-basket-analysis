"""Task A - classification: will the user reorder this product in the next order?"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import PLOT_COLOR, RANDOM_STATE
from .eda import save_fig
from .modeling import scaled


def candidate_models() -> list[tuple[str, str, object]]:
    """(family, config label, estimator) for every configuration tried."""
    models = []

    # l1_ratio=1 -> L1 (lasso) penalty, l1_ratio=0 -> L2 (ridge) penalty
    for label, l1_ratio in [('L1', 1), ('L2', 0)]:
        lr = LogisticRegression(
            C=0.001, l1_ratio=l1_ratio, class_weight='balanced',
            random_state=RANDOM_STATE, solver='liblinear',
        )
        models.append(('Logistic', label, scaled(lr, StandardScaler())))

    for k in [3, 5, 11]:
        models.append(('KNN', f'k={k}', scaled(KNeighborsClassifier(n_neighbors=k), MinMaxScaler())))

    for c in [0.001, 0.1]:
        svm = LinearSVC(C=c, class_weight='balanced', random_state=RANDOM_STATE)
        models.append(('LinearSVC', f'C={c}', scaled(svm, StandardScaler())))

    for c in [3, 10, 15, 21]:
        svm_k = SVC(C=c, kernel='rbf', class_weight='balanced', max_iter=10000)
        models.append(('SVM-Kernel', f'C={c}', scaled(svm_k, MinMaxScaler())))

    for depth in [8, 12]:
        dtc = DecisionTreeClassifier(
            max_depth=depth, min_samples_split=50, min_samples_leaf=40,
            class_weight='balanced', random_state=RANDOM_STATE,
        )
        models.append(('DecisionTree', f'depth={depth}', dtc))

    for n in [50, 100, 300]:
        for depth in [8, 12, 17]:
            rfc = RandomForestClassifier(
                n_estimators=n, max_depth=depth, min_samples_leaf=5, min_samples_split=9,
                class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
            )
            models.append(('RandomForest', f'n={n}, depth={depth}', rfc))

    lgbm = LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE, verbose=-1)
    models.append(('LightGBM', 'default', lgbm))

    return models


def score(model, X):
    """Positive-class probability, or the decision function for SVMs."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run(X_train, y_train, X_val, y_val, fig_dir: Path) -> pd.DataFrame:
    rows, roc_inputs = [], {}

    for family, label, model in candidate_models():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_score = score(model, X_val)

        row = {
            'family': family,
            'config': label,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_score),
            'pr_auc': average_precision_score(y_val, y_score),
        }
        rows.append(row)
        print(
            f'{family} {label} | Acc: {row["accuracy"]:.2f} | Precision: {row["precision"]:.2f} | '
            f'Recall: {row["recall"]:.2f} | F1: {row["f1"]:.2f} | '
            f'ROC-AUC: {row["roc_auc"]:.2f} | PR-AUC: {row["pr_auc"]:.2f}'
        )

        # Keep the best config per family (by PR-AUC: positives are only ~10%) for the ROC plot
        best = roc_inputs.get(family)
        if best is None or row['pr_auc'] > best[1]:
            roc_inputs[family] = (label, row['pr_auc'], y_score)

    plot_roc(y_val, roc_inputs, fig_dir)
    return pd.DataFrame(rows)


def plot_roc(y_val, roc_inputs: dict, fig_dir: Path) -> None:
    plt.figure(figsize=(12, 8))
    # fpr -> False Positive Rate (x, lower is better), tpr -> True Positive Rate (y, higher is better)
    for family, (label, _, y_score) in roc_inputs.items():
        fpr, tpr, _ = roc_curve(y_val, y_score)
        auc = roc_auc_score(y_val, y_score)
        plt.plot(fpr, tpr, label=f'{family} {label} (AUC={auc:.2f})')

    plt.plot([0, 1], [0, 1], '--', color=PLOT_COLOR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Config per Model Family')
    plt.legend()
    plt.grid(True)
    save_fig(fig_dir, '08_roc_task_a')
