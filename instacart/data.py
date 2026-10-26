from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ORDER_PRODUCTS_DTYPES, ORDERS_DTYPES


@dataclass
class RawData:
    aisles: pd.DataFrame
    departments: pd.DataFrame
    order_products_prior: pd.DataFrame
    order_products_train: pd.DataFrame
    orders: pd.DataFrame
    products: pd.DataFrame


def load_raw(data_dir: Path) -> RawData:
    """Read the six Instacart CSVs and cast text columns to category."""
    data_dir = Path(data_dir)

    aisles = pd.read_csv(data_dir / 'aisles.csv')
    departments = pd.read_csv(data_dir / 'departments.csv')
    order_products_prior = pd.read_csv(data_dir / 'order_products__prior.csv', dtype=ORDER_PRODUCTS_DTYPES)
    order_products_train = pd.read_csv(data_dir / 'order_products__train.csv', dtype=ORDER_PRODUCTS_DTYPES)
    orders = pd.read_csv(data_dir / 'orders.csv', dtype=ORDERS_DTYPES)
    products = pd.read_csv(data_dir / 'products.csv')

    aisles['aisle'] = aisles['aisle'].astype('category')
    departments['department'] = departments['department'].astype('category')
    products['product_name'] = products['product_name'].astype('category')

    return RawData(aisles, departments, order_products_prior, order_products_train, orders, products)


def build_prior_data(raw: RawData) -> pd.DataFrame:
    """One row per prior (order, product), joined with order, product, aisle and department info.

    validate='many_to_one' raises if a lookup key is duplicated, so a merge can never multiply rows.
    """
    orders_prior = raw.orders[raw.orders['eval_set'] == 'prior'].copy()
    return (
        raw.order_products_prior
        .merge(orders_prior, on='order_id', how='left', validate='many_to_one')
        .merge(raw.products, on='product_id', how='left', validate='many_to_one')
        .merge(raw.aisles, on='aisle_id', how='left', validate='many_to_one')
        .merge(raw.departments, on='department_id', how='left', validate='many_to_one')
    )


def load_prior_ids(data_dir: Path) -> pd.DataFrame:
    """Lightweight (user_id, product_id) view of the prior rows, used when features are reused from disk."""
    data_dir = Path(data_dir)
    prior = pd.read_csv(
        data_dir / 'order_products__prior.csv',
        usecols=['order_id', 'product_id'],
        dtype=ORDER_PRODUCTS_DTYPES,
    )
    orders = pd.read_csv(
        data_dir / 'orders.csv',
        usecols=['order_id', 'user_id'],
        dtype=ORDERS_DTYPES,
    )
    return prior.merge(orders, on='order_id', how='left')[['user_id', 'product_id']]
