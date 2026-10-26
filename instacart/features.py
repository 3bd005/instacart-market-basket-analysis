"""Feature engineering. Everything here is computed from prior orders only."""
import numpy as np
import pandas as pd


# 1) User-level features
def user_features(prior_data: pd.DataFrame) -> pd.DataFrame:
    user_orders = prior_data.groupby('user_id')['order_id'].nunique()
    user_orders.name = 'user_total_orders'

    basket_size = prior_data.groupby(['user_id', 'order_id'])['product_id'].count()
    user_avg_basket = basket_size.groupby('user_id').mean()
    user_avg_basket.name = 'user_avg_basket'

    user_reorder_ratio = prior_data.groupby('user_id')['reordered'].mean()
    user_reorder_ratio.name = 'user_reorder_ratio'

    user_mean_days = prior_data.groupby('user_id')['days_since_prior_order'].mean()
    user_mean_days.name = 'user_mean_days_between'

    user_last_recency = (
        prior_data.sort_values(['user_id', 'order_number'])
        .groupby('user_id')['days_since_prior_order'].last()
    )
    user_last_recency.name = 'user_last_order_recency'

    user_features_df = pd.concat(
        [user_orders, user_avg_basket, user_reorder_ratio, user_mean_days, user_last_recency],
        axis=1,
    )
    # 4) Engineered non-linear feature
    user_features_df['log_user_orders'] = np.log1p(user_features_df['user_total_orders'])
    return user_features_df


# 2) Product-level features
def product_features(prior_data: pd.DataFrame) -> pd.DataFrame:
    overall_reorder_rate = prior_data.groupby('product_id')['reordered'].mean()
    overall_reorder_rate.name = 'overall_reorder_rate'

    product_cart_pos = prior_data.groupby('product_id')['add_to_cart_order'].mean()
    product_cart_pos.name = 'product_avg_cart_position'

    product_popularity = prior_data.groupby('product_id')['order_id'].count()
    product_popularity.name = 'product_popularity_over_time_over_time'

    return pd.concat(
        [overall_reorder_rate, product_cart_pos, product_popularity],
        axis=1,
    ).reset_index()


# 3) User x product features
def user_product_features(prior_data: pd.DataFrame) -> pd.DataFrame:
    grouped = prior_data.groupby(['user_id', 'product_id'])

    user_product_prior_purchases = grouped.size()
    user_product_prior_purchases.name = 'user_product_prior_purchases'

    user_product_reorder_prob = grouped['reordered'].mean()
    user_product_reorder_prob.name = 'user_product_reorder_prob'

    user_product_avg_days_between = grouped['days_since_prior_order'].mean()
    user_product_avg_days_between.name = 'user_product_avg_days_between'

    user_product_reorder_count = grouped['reordered'].sum()
    user_product_reorder_count.name = 'reorder_count'

    return pd.concat(
        [
            user_product_prior_purchases,
            user_product_reorder_prob,
            user_product_avg_days_between,
            user_product_reorder_count,
        ],
        axis=1,
    ).reset_index()


# 5) Temporal features (one row per prior order)
def get_season(month: int) -> str:
    month = month % 12
    if month in [11, 0, 1]:
        return 'winter'
    if month in [2, 3, 4]:
        return 'spring'
    if month in [5, 6, 7]:
        return 'summer'
    return 'autumn'


def temporal_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Instacart has no dates, so a relative timeline is rebuilt from days_since_prior_order."""
    time_prior = orders.loc[
        orders['eval_set'] == 'prior',
        ['user_id', 'order_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'],
    ].sort_values(['user_id', 'order_number'])

    time_prior['order_hour'] = time_prior['order_hour_of_day']
    time_prior['order_day'] = time_prior['order_dow']
    time_prior['days_since_prior_order'] = time_prior['days_since_prior_order'].fillna(0)

    time_prior['cum_days'] = time_prior.groupby('user_id')['days_since_prior_order'].cumsum()
    time_prior['order_month'] = (time_prior['cum_days'] // 30).astype(int)
    time_prior['order_year'] = (time_prior['cum_days'] // 365).astype(int)
    time_prior['season'] = time_prior['order_month'].map(get_season)
    time_prior['is_weekend'] = time_prior['order_dow'].isin([0, 6]).astype(int)
    return time_prior


# 6) Aggregations over windows
def last_n_orders_window(time_prior: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Each user's last n prior orders, aggregated per user."""
    last_n = time_prior.groupby('user_id').tail(n)
    return last_n.groupby('user_id').agg(
        **{
            f'last{n}_mean_days_between': ('days_since_prior_order', 'mean'),
            f'last{n}_weekend_share': ('is_weekend', 'mean'),
            f'last{n}_mean_hour': ('order_hour_of_day', 'mean'),
        }
    )


# Train / test candidate tables: every product the user bought before, one row per (order, product)
def user_product_candidates(prior_data: pd.DataFrame) -> pd.DataFrame:
    candidates = prior_data.groupby('user_id')['product_id'].unique().explode().reset_index()
    candidates['product_id'] = candidates['product_id'].astype(np.int32)
    return candidates


def attach_features(
    df: pd.DataFrame,
    orders: pd.DataFrame,
    user_features_df: pd.DataFrame,
    product_features_df: pd.DataFrame,
    user_product_features_df: pd.DataFrame,
) -> pd.DataFrame:
    time = orders[['order_id', 'order_number', 'order_dow', 'order_hour_of_day']]
    return (
        df
        .merge(time, on='order_id', how='left', validate='many_to_one')
        .merge(user_features_df, on='user_id', how='left', validate='many_to_one')
        .merge(product_features_df, on='product_id', how='left', validate='many_to_one')
        .merge(user_product_features_df, on=['user_id', 'product_id'], how='left', validate='one_to_one')
    )


def build_train_df(
    candidates: pd.DataFrame,
    orders: pd.DataFrame,
    order_products_train: pd.DataFrame,
    *feature_tables: pd.DataFrame,
) -> pd.DataFrame:
    """Label = 1 if the previously bought product appears in the user's train order."""
    train_orders = orders[orders['eval_set'] == 'train'][['user_id', 'order_id']]

    train_df = train_orders.merge(candidates, on='user_id', how='left')
    train_df = train_df.merge(
        order_products_train[['order_id', 'product_id', 'reordered']],
        on=['order_id', 'product_id'],
        how='left',
        validate='one_to_one',
    )
    train_df['reordered'] = train_df['reordered'].fillna(0).astype(np.int8)
    return attach_features(train_df, orders, *feature_tables)


def build_test_df(
    candidates: pd.DataFrame,
    orders: pd.DataFrame,
    *feature_tables: pd.DataFrame,
) -> pd.DataFrame:
    test_orders = orders[orders['eval_set'] == 'test'][['user_id', 'order_id']]
    test_candidates = candidates[candidates['user_id'].isin(test_orders['user_id'])]

    test_df = test_orders.merge(test_candidates, on='user_id', how='left')
    test_df = attach_features(test_df, orders, *feature_tables)
    return test_df.drop(columns=['reorder_count'])
