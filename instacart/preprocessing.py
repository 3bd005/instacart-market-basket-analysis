import numpy as np
import pandas as pd

# Ids, codes and the sentinel-filled column are not checked for outliers
NOT_OUTLIER_CANDIDATES = [
    'order_id', 'user_id', 'product_id', 'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day', 'days_since_prior_order',
]


def fill_days_since_prior(prior_data: pd.DataFrame) -> pd.DataFrame:
    """A user's first order has no previous order: mark it with the sentinel -1."""
    prior_data['days_since_prior_order'] = prior_data['days_since_prior_order'].fillna(-1)
    return prior_data


def find_outlier_cols(prior_data: pd.DataFrame) -> list[str]:
    numerical_cols = prior_data.select_dtypes(include=[np.number]).columns
    return [
        col for col in numerical_cols
        if col not in NOT_OUTLIER_CANDIDATES and prior_data[col].nunique() > 2
    ]


def clip_outliers_iqr(prior_data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]: reduces outlier influence but keeps every row."""
    for col in cols:
        q1, q3 = prior_data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        prior_data[col] = prior_data[col].clip(lower=lower, upper=upper)
    return prior_data
