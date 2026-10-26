import numpy as np

RANDOM_STATE = 42

# Read dtypes (keep the 32M-row prior table small in memory)
ORDER_PRODUCTS_DTYPES = {
    'order_id': np.int32,
    'product_id': np.int32,
    'add_to_cart_order': np.int16,
    'reordered': np.int8,
}
ORDERS_DTYPES = {
    'order_id': np.int32,
    'user_id': np.int64,
    'eval_set': 'category',
    'order_number': np.int16,
    'order_dow': np.int8,
    'order_hour_of_day': np.int8,
    'days_since_prior_order': np.float32,
}

# Low-cardinality categoricals -> one-hot
LOW_CAT_COLS = ['order_dow', 'order_hour_of_day']

# High-cardinality ids -> K-fold target encoding
TE_SOURCE_COLS = ['user_id', 'product_id']
TE_COLS = ['te_user_id', 'te_product_id']

# Continuous numeric features (scaled for linear / distance models, checked with VIF)
COLS_SCALING = [
    'user_total_orders', 'user_avg_basket', 'user_reorder_ratio',
    'user_mean_days_between', 'user_last_order_recency', 'log_user_orders',
    'overall_reorder_rate', 'product_avg_cart_position',
    'product_popularity_over_time_over_time',
    'user_product_prior_purchases', 'user_product_reorder_prob',
    'user_product_avg_days_between',
]

# Columns that are never model inputs (ids, targets, target encodings)
COLS_DROP = [
    'reordered',
    'order_id',
    'user_id',
    'product_id',
    'order_number',
    'te_user_id',
    'te_product_id',
    'reorder_count',
]

TARGET_A = 'reordered'      # Task A: classification
TARGET_B = 'reorder_count'  # Task B: regression

PLOT_COLOR = '#0073e6'
