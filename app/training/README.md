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

The script writes:

- `app/models/forecast_model.joblib`
- `app/models/model_metadata.json`

If LightGBM is available, the model version is `lightgbm-window-v1`. Otherwise
the script tries XGBoost, then scikit-learn HistGradientBoostingRegressor.
