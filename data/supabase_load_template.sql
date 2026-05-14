-- Previzma B2B dataset load template
--
-- Generate CSVs first:
-- python -m app.training.transform_kaggle_b2b --input data/raw/online_retail_II.csv --output-dir data/processed
--
-- Supabase SQL Editor does not support \copy.
-- Use these commands with psql from the project root, or import the CSV files
-- through Supabase Table Editor in this order:
-- 1. data/processed/products.csv
-- 2. data/processed/client_segments.csv
-- 3. data/processed/sales.csv

\copy products(id, name, sku, description, status, company_id) from 'data/processed/products.csv' csv header;
\copy client_segments(id, name, type, description, active, company_id) from 'data/processed/client_segments.csv' csv header;
\copy sales(sale_date, quantity, amount, confirmed_order, source_status, product_id, client_segment_id) from 'data/processed/sales.csv' csv header;

-- Canonical ML export after loading Supabase with real IDs:
select
  s.sale_date as "saleDate",
  p.id as "productId",
  p.name as "productName",
  p.sku as "productSku",
  cs.id as "clientSegmentId",
  cs.name as "clientSegmentName",
  cs.type as "clientSegmentType",
  s.quantity,
  s.amount,
  s.confirmed_order as "confirmedOrder",
  s.source_status as "sourceStatus"
from sales s
join products p on p.id = s.product_id
join client_segments cs on cs.id = s.client_segment_id
where p.company_id = 1
  and cs.company_id = 1
order by s.sale_date;
