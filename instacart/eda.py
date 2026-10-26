from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # write PNGs, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PLOT_COLOR


def save_fig(fig_dir: Path, name: str) -> None:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f'{name}.png', bbox_inches='tight', dpi=110)
    plt.close('all')


def plot_missing_values(prior_data: pd.DataFrame, fig_dir: Path) -> None:
    columns_with_na = prior_data.isnull().sum()
    columns_with_na.plot(kind='bar', color=PLOT_COLOR, figsize=(8, 5), width=.2, edgecolor='black')
    plt.xlabel('Columns with Missing Values')
    plt.ylabel('Number of Missing Values')
    plt.title('Missing Values in prior_data')
    plt.xticks(rotation=45)
    save_fig(fig_dir, '01_missing_values')


def plot_numerical_distributions(prior_data: pd.DataFrame, fig_dir: Path) -> None:
    numerical_cols = prior_data.select_dtypes(include=[np.number]).columns
    prior_data[numerical_cols].hist(bins=15, color=PLOT_COLOR, edgecolor='black', figsize=(15, 10))
    plt.suptitle('Distribution of Numerical Features', fontsize=16)
    save_fig(fig_dir, '02_numerical_distributions')


def plot_categorical_counts(orders: pd.DataFrame, prior_data: pd.DataFrame, fig_dir: Path) -> None:
    orders['eval_set'].value_counts().plot(kind='bar', color=PLOT_COLOR, width=.2, figsize=(10, 7))
    plt.xlabel('eval_set')
    plt.ylabel('Count')
    plt.title('Orders per eval_set')
    plt.xticks(rotation=45)
    save_fig(fig_dir, '03_orders_eval_set')

    categorical_cols = prior_data.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        prior_data[col].value_counts().head().plot(kind='bar', color=PLOT_COLOR)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Top 5 Categories in {col}')
        plt.xticks(rotation=45)
        save_fig(fig_dir, f'04_top5_{col}')


def plot_correlation(prior_data: pd.DataFrame, fig_dir: Path) -> None:
    numerical_cols = prior_data.select_dtypes(include=[np.number]).columns
    corr = prior_data[numerical_cols].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features')
    save_fig(fig_dir, '05_correlation_matrix')


def plot_order_patterns(prior_data: pd.DataFrame, fig_dir: Path) -> None:
    patterns = [
        ('order_hour_of_day', 'Hour of Day', (10, 6)),
        ('order_dow', 'Day of Week', (10, 6)),
        ('order_number', 'Order Number', (12, 7)),
    ]
    for col, label, size in patterns:
        plt.figure(figsize=size)
        sns.countplot(data=prior_data, x=col, hue=col, palette='viridis', legend=False)
        plt.title(f'Products Ordered by {label}')
        plt.xlabel(label)
        plt.ylabel('Number of Products Ordered')
        save_fig(fig_dir, f'06_count_by_{col}')


def plot_boxplots(prior_data: pd.DataFrame, cols: list[str], stage: str, fig_dir: Path) -> None:
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=prior_data, x=col)
        plt.title(f'Boxplot of {col} {stage} outlier clipping')
        save_fig(fig_dir, f'07_boxplot_{col}_{stage.lower()}')


def run_eda(prior_data: pd.DataFrame, orders: pd.DataFrame, fig_dir: Path) -> None:
    plot_missing_values(prior_data, fig_dir)
    plot_numerical_distributions(prior_data, fig_dir)
    plot_categorical_counts(orders, prior_data, fig_dir)
    plot_correlation(prior_data, fig_dir)
    plot_order_patterns(prior_data, fig_dir)
