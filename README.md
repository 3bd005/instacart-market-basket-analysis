# Instacart Reorder Prediction – Machine Learning Project

An end-to-end machine learning pipeline on the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset: data loading, EDA, preprocessing, feature engineering, encoding, class-imbalance handling, and a comparison of classification and regression models.

## Contributions

- **Abdullah:** Worked on the entire project.
- **Ibrahim:** Worked on the README and the baseline for Task B.

## Tasks

| Task | Type | Question | Target |
|---|---|---|---|
| A | Classification | Will the user reorder a product they bought before in their next order? | `reordered` (0/1, about 9.8% positives) |
| B | Regression | How many times has the user reordered this product? | `reorder_count` |

## Dataset

Six CSV files from Instacart: `aisles.csv`, `departments.csv`, `products.csv`, `orders.csv`, `order_products__prior.csv` and `order_products__train.csv`.

- Prior orders (about 32.4M order–product rows) are used **only** to build features.
- The user's next order (`eval_set = train`) gives the label.
- After feature engineering: `train_df` has about 8.47M rows, and `test_df` has about 4.83M rows.

## Project structure

```
ML_project_main_file_final_run.ipynb   # original Colab notebook
main.py                                # runs the whole pipeline, step by step
pyproject.toml                         # dependencies
instacart/
    config.py          # constants: dtypes, feature lists, columns to drop
    data.py            # load the CSVs, build prior_data
    eda.py             # EDA figures (saved as PNG)
    preprocessing.py   # missing-value sentinel, IQR outlier clipping
    features.py        # user / product / user x product / temporal / window features, train & test tables
    encoding.py        # one-hot, K-fold target encoding, frequency encoding, VIF
    modeling.py        # time-aware split, scaling pipelines, imbalance experiment
    task_a.py          # Task A: classification models
    task_b.py          # Task B: regression models
```

## How to run

1. Download the six CSVs from Kaggle and put them in a `data/` folder.
2. Run with [uv](https://docs.astral.sh/uv/), which installs the dependencies from `pyproject.toml`:

```
uv run main.py --data-dir data --out-dir outputs
```

| Option | Effect |
|---|---|
| `--skip-eda` | Skip the EDA figures. |
| `--reuse-features` | Load `outputs/train_df.csv` and `outputs/test_df.csv` from an earlier run instead of rebuilding them from the 32M-row prior table. |
| `--sample-size N` | Rows sampled for the model experiments (default 200000). |

Results are written to `outputs/`: `figures/*.png`, `train_df.csv`, `test_df.csv`, `task_a_results.csv` and `task_b_results.csv`.
The full dataset needs several GB of RAM for the prior-order merge.

To use the notebook instead, open it in Google Colab and put the CSVs in `MyDrive/project files/`.

## Pipeline

### 1. Data loading
The CSVs are read with compact dtypes (`int8`/`int16`/`int32`, `category`). Prior orders are merged with orders, products, aisles and departments into one `prior_data` table.

### 2. Exploratory data analysis
- Missing values per column
- Distributions of numerical features
- Top categories (eval set, product, aisle, department)
- Correlation heatmap
- Order patterns by hour of day, day of week and order number

### 3. Missing values and outliers
- `days_since_prior_order` is empty for a user's first order, so it is filled with the sentinel `-1`.
- `add_to_cart_order` and `order_number` are clipped to `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`. This reduces the effect of outliers while keeping every row.

### 4. Feature engineering (from prior orders only)

| Level | Features |
|---|---|
| User | total orders, average basket size, reorder ratio, mean days between orders, recency of last order |
| Product | overall reorder rate, average add-to-cart position, popularity |
| User × product | times purchased, reorder probability, average days between purchases, reorder count |
| Non-linear | `log_user_orders = log1p(user_total_orders)` |
| Temporal | relative month, year and season rebuilt from cumulative days, weekend flag |
| Window | mean days between orders, weekend share and mean hour over each user's last 3 orders |

Training rows are every product a user bought in prior orders, paired with that user's next order. The label is 1 if the product appears in that next order.

### 5. Encoding
- **One-hot:** `order_dow`, `order_hour_of_day`.
- **Target (mean) encoding:** `user_id`, `product_id`. It is computed out-of-fold with 5-fold stratified K-fold and smoothing of 50, so a row's own label never leaks into its encoding.
- **Frequency encoding:** `user_id`, `product_id`.

The two id encodings are compared by the ROC-AUC of a 5-fold logistic regression, and target encoding is kept.

### 6. Multicollinearity
The variance inflation factor (VIF) is computed for the 12 continuous features. Features with a VIF above 10 are flagged for removal.

### 7. Validation
A random split would let the model learn from later orders. Instead, the validation is time-aware:
1. Sample 200,000 rows.
2. Train on orders at or below the 75th percentile of `order_number`.
3. Validate on later orders.

Linear, SVM and KNN models scale the continuous features (`StandardScaler` or `MinMaxScaler`) inside a pipeline. Tree models use the raw values.

### 8. Class imbalance
Four strategies are compared with logistic regression on the validation set: no correction, class weights, SMOTE and random undersampling. All Task A models except KNN, which has no class-weight option, use `class_weight='balanced'`.

### 9. Models

| Task A – classification | Task B – regression |
|---|---|
| Logistic Regression (L1, L2) | Linear Regression (OLS), Ridge, Lasso, Elastic Net |
| KNN | KNN (Euclidean, Manhattan) |
| Linear SVM, SVM (RBF kernel) | Linear SVR, SVR (RBF kernel) |
| Decision Tree | Decision Tree |
| Random Forest | Random Forest |
| LightGBM | LightGBM (pipeline) |

### 10. Evaluation metrics
- **Task A:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and a ROC curve for the best setting of each model.
- **Task B:** MAE, MSE, RMSE, R².

## Notes

- Ids, target-encoded ids and both targets are excluded from model inputs.
- The temporal and window features are computed and reported, but they are not model inputs.
