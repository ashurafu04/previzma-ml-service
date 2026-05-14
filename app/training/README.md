# Offline Forecast Training

FastAPI remains stateless at runtime. Training starts from a CSV export prepared
outside this service, typically from Supabase SQL Editor or Spring-managed data
exports.

Expected CSV columns:

```text
saleDate,productId,productName,productSku,clientSegmentId,clientSegmentName,clientSegmentType,quantity,amount,confirmedOrder,sourceStatus
```

Example Supabase export query:

```sql
select
  s.sale_date as "saleDate",
  p.id as "productId",
  p.name as "productName",
  p.sku as "productSku",
  cs.id as "clientSegmentId",
  cs.name as "clientSegmentName",
  cs.type as "clientSegmentType",
  s.quantity as "quantity",
  s.amount as "amount",
  s.confirmed_order as "confirmedOrder",
  s.source_status as "sourceStatus"
from sales s
join products p on p.id = s.product_id
join client_segments cs on cs.id = s.client_segment_id
where s.sale_date is not null;
```

Export the result as CSV and train locally:

```powershell
python -m app.training.train_model --input data/sales_export.csv --output app/models/forecast_model.joblib
```

For Kaggle demo data, run the transformation first:

```powershell
python -m app.training.transform_kaggle_b2b --input data/raw/online_retail_II.csv --output-dir data/processed --max-output-rows 30000
```

Then load the generated `products.csv`, `client_segments.csv`, and `sales.csv`
into Supabase, re-export the canonical ML CSV with real IDs, and train from that
export.

The script writes:

- `app/models/forecast_model.joblib`
- `app/models/model_metadata.json`

If LightGBM is available, the model version is `lightgbm-window-v1`. Otherwise
the script tries XGBoost, then scikit-learn HistGradientBoostingRegressor.

The training script compares target stabilization strategies on the same
temporal validation split:

- `raw`: train directly on future revenue
- `log1p`: train on `log1p(future revenue)`, then predict with `expm1`
- `clipped_raw`: winsorize extreme targets before training
- `clipped_log1p`: winsorize extreme targets, then train on `log1p`
- `baseline_ratio_log1p`: train on `log1p(actual / baseline_prediction)`,
  then multiply the inverse prediction by the statistical baseline
- `clipped_baseline_ratio_log1p`: clip extreme baseline ratios before the
  `log1p` transform

Validation MAE, MAPE, and RMSE are always computed against the true, unmodified
future revenue in MAD. The best strategy is selected by lowest validation MAPE,
then RMSE. `model_metadata.json` records the selected `targetStrategy` and all
`targetStrategyCandidates`.
