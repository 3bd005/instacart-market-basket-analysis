import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import COLS_SCALING, LOW_CAT_COLS, RANDOM_STATE, TARGET_A, TE_COLS, TE_SOURCE_COLS


def one_hot_encode(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_train = ohe.fit_transform(train_df[LOW_CAT_COLS])
    ohe_test = ohe.transform(test_df[LOW_CAT_COLS])

    feature_names = ohe.get_feature_names_out(LOW_CAT_COLS)
    train_ohe_df = pd.DataFrame(ohe_train, columns=feature_names, index=train_df.index)
    test_ohe_df = pd.DataFrame(ohe_test, columns=feature_names, index=test_df.index)

    train_df = pd.concat([train_df.drop(columns=LOW_CAT_COLS), train_ohe_df], axis=1)
    test_df = pd.concat([test_df.drop(columns=LOW_CAT_COLS), test_ohe_df], axis=1)
    return train_df, test_df


def target_encode_kfold(train_df: pd.DataFrame, test_df: pd.DataFrame, smoothing: float = 50) -> None:
    """Out-of-fold target encoding on train (no leakage); test uses an encoder fit on all of train."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_df[TE_COLS] = 0.0

    for tr_idx, val_idx in skf.split(train_df, train_df[TARGET_A]):
        encoder = TargetEncoder(cols=TE_SOURCE_COLS, smoothing=smoothing)
        encoder.fit(train_df.iloc[tr_idx][TE_SOURCE_COLS], train_df.iloc[tr_idx][TARGET_A])
        te = encoder.transform(train_df.iloc[val_idx][TE_SOURCE_COLS])
        te.columns = TE_COLS
        train_df.loc[train_df.index[val_idx], TE_COLS] = te

    final_encoder = TargetEncoder(cols=TE_SOURCE_COLS, smoothing=smoothing)
    final_encoder.fit(train_df[TE_SOURCE_COLS], train_df[TARGET_A])
    te_test = final_encoder.transform(test_df[TE_SOURCE_COLS])
    te_test.columns = TE_COLS
    test_df[TE_COLS] = te_test


def frequency_maps(prior_ids: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Share of prior rows per product and per user (normalize=True -> fractions)."""
    return (
        prior_ids['product_id'].value_counts(normalize=True),
        prior_ids['user_id'].value_counts(normalize=True),
    )


def compare_encodings(train_df: pd.DataFrame, product_freq: pd.Series, user_freq: pd.Series) -> dict[str, float]:
    """5-fold ROC-AUC of a logistic regression on target-encoded vs frequency-encoded ids."""
    freq = pd.DataFrame({
        'product_id_freq': train_df['product_id'].map(product_freq).fillna(0.0),
        'user_id_freq': train_df['user_id'].map(user_freq).fillna(0.0),
    })

    lg = LogisticRegression(class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = train_df[TARGET_A]
    return {
        'target': cross_val_score(lg, train_df[TE_COLS], y, cv=cv, scoring='roc_auc').mean(),
        'frequency': cross_val_score(lg, freq, y, cv=cv, scoring='roc_auc').mean(),
    }


def vif_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor of each continuous feature (computed on standardised values)."""
    x_vif = pd.DataFrame(
        StandardScaler().fit_transform(train_df[COLS_SCALING]),
        columns=COLS_SCALING,
    )
    x_vif = x_vif.replace([np.inf, -np.inf], np.nan).dropna()
    return pd.DataFrame({
        'feature': x_vif.columns,
        'VIF_value': [variance_inflation_factor(x_vif.values, i) for i in range(x_vif.shape[1])],
    }).sort_values(by='VIF_value')
