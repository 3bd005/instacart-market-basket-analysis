"""Shared modelling helpers: time-aware split, scaling pipelines, imbalance experiment."""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from .config import COLS_DROP, COLS_SCALING, RANDOM_STATE


def time_aware_split(
    train_df: pd.DataFrame,
    target: str,
    sample_size: int,
    quantile: float = .75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train on earlier orders, validate on later ones (cut on order_number), never a random split."""
    sample_df = train_df.sample(n=sample_size, random_state=RANDOM_STATE)
    df = sample_df.sort_values(['user_id', 'order_number'])

    cut = df['order_number'].quantile(quantile)
    train_split = df[df['order_number'] <= cut]
    val_split = df[df['order_number'] > cut]

    return (
        train_split.drop(columns=COLS_DROP),
        val_split.drop(columns=COLS_DROP),
        train_split[target],
        val_split[target],
    )


def scaled(model, scaler) -> Pipeline:
    """Scale the continuous features only; one-hot columns pass through unchanged."""
    preprocess = ColumnTransformer(
        [('scale', scaler, COLS_SCALING)],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    return Pipeline([('preprocess', preprocess), ('model', model)])


def imbalance_experiment(X_train, y_train, X_val, y_val) -> None:
    """Logistic regression with and without imbalance corrections."""
    x_smote, y_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    x_under, y_under = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    methods = [
        ('Baseline', X_train, y_train, LogisticRegression()),
        ('Class Weights', X_train, y_train, LogisticRegression(class_weight='balanced')),
        ('SMOTE', x_smote, y_smote, LogisticRegression()),
        ('Undersampling', x_under, y_under, LogisticRegression()),
    ]
    for name, x_tr, y_tr, model in methods:
        model.fit(x_tr, y_tr)
        print(f'{name} :')
        print(classification_report(y_val, model.predict(X_val), zero_division=0))
