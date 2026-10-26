"""Task B - regression: how many times has the user reordered this product?"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_STATE
from .modeling import scaled


def candidate_models() -> list[tuple[str, str, object]]:
    """(family, config label, estimator) for every configuration tried."""
    models = [
        ('OLS', 'default', scaled(LinearRegression(), StandardScaler())),
        ('Ridge', 'alpha=11', scaled(Ridge(alpha=11, random_state=RANDOM_STATE), StandardScaler())),
        ('Lasso', 'alpha=0.7', scaled(Lasso(alpha=0.7, random_state=RANDOM_STATE), StandardScaler())),
        ('ElasticNet', 'alpha=1, l1_ratio=0.5',
         scaled(ElasticNet(alpha=1, l1_ratio=0.5, random_state=RANDOM_STATE), StandardScaler())),
        ('LinearSVR', 'C=0.0001', scaled(LinearSVR(C=0.0001, random_state=RANDOM_STATE), StandardScaler())),
        ('SVR-Kernel', 'C=10', scaled(SVR(C=10, kernel='rbf', max_iter=8000), StandardScaler())),
    ]

    # p=2 Euclidean, p=1 Manhattan
    for k in [5, 11, 25, 35]:
        for dist, p in [('euclidean', 2), ('manhattan', 1)]:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance', p=p)
            models.append(('KNN', f'k={k}, dist={dist}', scaled(knn, StandardScaler())))

    models += [
        ('DecisionTree', 'depth=3', DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)),
        ('RandomForest', 'n=300, depth=8', RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1,
        )),
        ('LightGBM', 'pipeline', Pipeline([('lgbm', LGBMRegressor(verbose=-1))])),
    ]
    return models


def run(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    rows = []
    for family, label, model in candidate_models():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        row = {
            'family': family,
            'config': label,
            'mae': mean_absolute_error(y_val, y_pred),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2_score(y_val, y_pred),
        }
        rows.append(row)
        print(
            f'{family} {label} | MAE={row["mae"]:.2f} | MSE={row["mse"]:.2f} | '
            f'RMSE={row["rmse"]:.2f} | R2={row["r2"]:.2f}'
        )
    return pd.DataFrame(rows)
