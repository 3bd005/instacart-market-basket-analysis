"""Instacart reorder prediction - end-to-end pipeline.

    uv run main.py --data-dir data --out-dir outputs
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from instacart import encoding, features, preprocessing, task_a, task_b
from instacart.config import TARGET_A, TARGET_B
from instacart.data import build_prior_data, load_prior_ids, load_raw
from instacart.eda import plot_boxplots, run_eda
from instacart.modeling import imbalance_experiment, time_aware_split

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 30)


def section(title: str) -> None:
    print(f'\n{"=" * 25} {title} {"=" * 25}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='folder with the six Instacart CSVs (default: data)')
    parser.add_argument('--out-dir', type=Path, default=Path('outputs'),
                        help='where figures, feature tables and results are written (default: outputs)')
    parser.add_argument('--sample-size', type=int, default=200_000,
                        help='rows sampled from train_df for the model experiments (default: 200000)')
    parser.add_argument('--skip-eda', action='store_true', help='do not draw the EDA figures')
    parser.add_argument('--reuse-features', action='store_true',
                        help='load out-dir/train_df.csv and test_df.csv instead of rebuilding them')
    return parser.parse_args()


def build_features(args, fig_dir: Path, train_csv: Path, test_csv: Path):
    section('Load data')
    raw = load_raw(args.data_dir)
    prior_data = build_prior_data(raw)
    print(f'prior_data: {prior_data.shape}')

    if not args.skip_eda:
        section('EDA')
        run_eda(prior_data, raw.orders, fig_dir)

    section('Missing values & outliers')
    preprocessing.fill_days_since_prior(prior_data)
    outlier_cols = preprocessing.find_outlier_cols(prior_data)
    print(f'outlier columns: {outlier_cols}')
    if not args.skip_eda:
        plot_boxplots(prior_data, outlier_cols, 'Before', fig_dir)
    preprocessing.clip_outliers_iqr(prior_data, outlier_cols)
    if not args.skip_eda:
        plot_boxplots(prior_data, outlier_cols, 'After', fig_dir)

    section('Feature engineering')
    user_features_df = features.user_features(prior_data)
    product_features_df = features.product_features(prior_data)
    user_product_features_df = features.user_product_features(prior_data)
    print(user_features_df.head(), '\n')
    print(product_features_df.head(), '\n')
    print(user_product_features_df.head(), '\n')

    time_prior = features.temporal_features(raw.orders)
    print('season share:', time_prior['season'].value_counts(normalize=True).round(3).to_dict())
    print('weekend share:', round(time_prior['is_weekend'].mean(), 3))
    print(features.last_n_orders_window(time_prior, n=3).head(), '\n')

    feature_tables = (user_features_df, product_features_df, user_product_features_df)
    candidates = features.user_product_candidates(prior_data)
    train_df = features.build_train_df(candidates, raw.orders, raw.order_products_train, *feature_tables)
    test_df = features.build_test_df(candidates, raw.orders, *feature_tables)
    print(f'train_df: {train_df.shape} | missing: {int(train_df.isnull().sum().sum())}')
    print(f'test_df:  {test_df.shape} | missing: {int(test_df.isnull().sum().sum())}')

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f'saved {train_csv} and {test_csv}')

    freq_maps = encoding.frequency_maps(prior_data[['user_id', 'product_id']])
    return train_df, test_df, freq_maps


def main() -> None:
    args = parse_args()
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / 'figures'
    train_csv = args.out_dir / 'train_df.csv'
    test_csv = args.out_dir / 'test_df.csv'

    if args.reuse_features and train_csv.exists() and test_csv.exists():
        section('Load saved features')
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        freq_maps = encoding.frequency_maps(load_prior_ids(args.data_dir))
        print(f'train_df: {train_df.shape} | test_df: {test_df.shape}')
    else:
        train_df, test_df, freq_maps = build_features(args, fig_dir, train_csv, test_csv)

    section('Encoding')
    train_df, test_df = encoding.one_hot_encode(train_df, test_df)
    encoding.target_encode_kfold(train_df, test_df)
    roc = encoding.compare_encodings(train_df, *freq_maps)
    print(f'Target Encoding ROC-AUC: {roc["target"]:.2f}')
    print(f'Frequency Encoding ROC-AUC: {roc["frequency"]:.2f}')
    print('-> target (mean) encoding kept')

    section('Multicollinearity (VIF)')
    vif_df = encoding.vif_table(train_df)
    print(vif_df.to_string(index=False))
    high_vif = vif_df.loc[vif_df['VIF_value'] > 10, 'feature'].tolist()
    print(f'features with VIF > 10: {high_vif or "none"}')

    section('Class imbalance')
    pers_one = train_df[TARGET_A].mean()
    print(f'reordered=1: {pers_one:.4f} | reordered=0: {1 - pers_one:.4f}')
    split_a = time_aware_split(train_df, TARGET_A, args.sample_size)
    print(f'time-aware split: train={len(split_a[0])} rows, val={len(split_a[1])} rows')
    X_train, X_val, y_train, y_val = split_a
    imbalance_experiment(X_train, y_train, X_val, y_val)

    section('Task A: classification (reordered)')
    results_a = task_a.run(X_train, y_train, X_val, y_val, fig_dir)
    results_a.to_csv(args.out_dir / 'task_a_results.csv', index=False)

    section('Task B: regression (reorder_count)')
    X_train, X_val, y_train, y_val = time_aware_split(train_df, TARGET_B, args.sample_size)
    results_b = task_b.run(X_train, y_train, X_val, y_val)
    results_b.to_csv(args.out_dir / 'task_b_results.csv', index=False)

    section('Summary')
    print('Task A - best config per family (by PR-AUC):')
    best_a = results_a.loc[results_a.groupby('family', sort=False)['pr_auc'].idxmax()]
    print(best_a.round(3).to_string(index=False))
    print('\nTask B - best config per family (by RMSE):')
    best_b = results_b.loc[results_b.groupby('family', sort=False)['rmse'].idxmin()]
    print(best_b.round(3).to_string(index=False))
    print(f'\nresults and figures in {args.out_dir.resolve()} ({(time.time() - start) / 60:.1f} min)')


if __name__ == '__main__':
    main()
