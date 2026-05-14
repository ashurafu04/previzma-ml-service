from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ProductDefinition:
    id: int
    name: str
    sku: str
    description: str
    unit_price_mad: float
    quantity_scale: float
    max_quantity: int


@dataclass(frozen=True)
class SegmentDefinition:
    id: int
    name: str
    type: str
    description: str
    multiplier: float


PRODUCTS = (
    ProductDefinition(1, "Industrial Pump X200", "PRD-PUMP-X200", "Pompe industrielle haute pression pour lignes de production.", 52000, 0.16, 80),
    ProductDefinition(2, "Smart Conveyor S400", "PRD-CONV-S400", "Convoyeur intelligent pour usines et plateformes logistiques.", 180000, 0.06, 25),
    ProductDefinition(3, "Hydraulic Valve HV90", "PRD-HV90", "Valve hydraulique de precision pour equipements industriels.", 32000, 0.22, 120),
    ProductDefinition(4, "Packaging Robot PR300", "PRD-ROBOT-PR300", "Robot de packaging pour chaines industrielles automatisees.", 420000, 0.035, 8),
    ProductDefinition(5, "Air Compressor AC700", "PRD-COMP-AC700", "Compresseur d'air industriel pour ateliers de production.", 95000, 0.09, 40),
    ProductDefinition(6, "Industrial Sensor IS50", "PRD-SENSOR-IS50", "Capteur industriel pour supervision et maintenance predictive.", 4500, 1.8, 500),
    ProductDefinition(7, "Control Panel CP120", "PRD-PANEL-CP120", "Armoire de controle pour machines et lignes automatisees.", 65000, 0.13, 60),
    ProductDefinition(8, "Spare Parts Kit SPK", "PRD-SPK", "Kit de pieces de rechange pour maintenance industrielle.", 12000, 0.75, 260),
)

SEGMENTS = (
    SegmentDefinition(1, "Grand comptes Maroc", "GRAND_COMPTE", "Comptes industriels a volume recurrent au Maroc.", 1.18),
    SegmentDefinition(2, "PME industrielles", "PME", "PME industrielles avec commandes recurrentes moderees.", 0.86),
    SegmentDefinition(3, "Distributeurs Afrique du Nord", "DISTRIBUTEUR", "Distributeurs regionaux servant l'Afrique du Nord.", 1.05),
)

SEASONAL_FACTORS = {
    1: 0.92,
    2: 0.96,
    3: 1.05,
    4: 1.08,
    5: 1.02,
    6: 0.98,
    7: 0.88,
    8: 0.84,
    9: 1.08,
    10: 1.16,
    11: 1.22,
    12: 1.18,
}

ENRICHED_COLUMNS = [
    "saleDate",
    "productSku",
    "productName",
    "clientSegmentName",
    "clientSegmentType",
    "quantity",
    "amount",
    "confirmedOrder",
    "sourceStatus",
]

SALES_EXPORT_COLUMNS = [
    "saleDate",
    "productId",
    "productName",
    "productSku",
    "clientSegmentId",
    "clientSegmentName",
    "clientSegmentType",
    "quantity",
    "amount",
    "confirmedOrder",
    "sourceStatus",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transform Online Retail II Kaggle data into Previzma B2B datasets."
    )
    parser.add_argument("--input", default="data/raw/online_retail_II.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--sales-export-output",
        default=None,
        help="Optional path for the ML sales_export.csv. Defaults to output-dir/sales_export.csv.",
    )
    parser.add_argument("--company-id", type=int, default=1)
    parser.add_argument("--max-output-rows", type=int, default=20000)
    parser.add_argument("--target-start", default="2021-01-01")
    parser.add_argument("--target-end", default="2026-12-31")
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=None,
        help="Optional development limit for reading a small raw sample.",
    )
    args = parser.parse_args(argv)

    output_paths = transform_kaggle_b2b(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        sales_export_output=Path(args.sales_export_output)
        if args.sales_export_output
        else None,
        company_id=args.company_id,
        max_output_rows=args.max_output_rows,
        target_start=date.fromisoformat(args.target_start),
        target_end=date.fromisoformat(args.target_end),
        max_input_rows=args.max_input_rows,
    )

    print(json.dumps({key: str(value) for key, value in output_paths.items()}, indent=2))
    return 0


def transform_kaggle_b2b(
    input_path: Path,
    output_dir: Path,
    sales_export_output: Path | None = None,
    company_id: int = 1,
    max_output_rows: int = 20000,
    target_start: date = date(2021, 1, 1),
    target_end: date = date(2026, 12, 31),
    max_input_rows: int | None = None,
) -> dict[str, Path]:
    raw_df = pd.read_csv(input_path, nrows=max_input_rows, low_memory=False)
    normalized_df = _normalize_raw_columns(raw_df)
    enriched_df = _build_enriched_sales(
        normalized_df,
        max_output_rows=max_output_rows,
        target_start=target_start,
        target_end=target_end,
    )

    products_df = _products_dataframe(company_id)
    segments_df = _segments_dataframe(company_id)
    sales_df = _sales_dataframe(enriched_df)
    sales_export_df = _sales_export_dataframe(enriched_df)
    summary = _dataset_summary(sales_export_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    sales_export_path = sales_export_output or output_dir / "sales_export.csv"
    sales_export_path.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "products": output_dir / "products.csv",
        "client_segments": output_dir / "client_segments.csv",
        "sales": output_dir / "sales.csv",
        "enriched_sales": output_dir / "previzma_b2b_sales_enriched.csv",
        "sales_export": sales_export_path,
        "summary": output_dir / "dataset_summary.json",
        "load_sql": output_dir / "load_supabase.sql",
    }

    products_df.to_csv(paths["products"], index=False)
    segments_df.to_csv(paths["client_segments"], index=False)
    sales_df.to_csv(paths["sales"], index=False)
    enriched_df[ENRICHED_COLUMNS].to_csv(paths["enriched_sales"], index=False)
    sales_export_df[SALES_EXPORT_COLUMNS].to_csv(paths["sales_export"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["load_sql"].write_text(_load_sql_template(paths), encoding="utf-8")

    return paths


def _normalize_raw_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "invoice": ("invoice", "invoiceno"),
        "stock_code": ("stockcode",),
        "description": ("description",),
        "quantity": ("quantity",),
        "invoice_date": ("invoicedate",),
        "unit_price": ("unitprice", "price"),
        "customer_id": ("customerid",),
        "country": ("country",),
    }
    normalized_lookup = {_normalize_name(column): column for column in raw_df.columns}
    rename_map = {}
    missing = []

    for canonical_name, candidates in aliases.items():
        source_column = next(
            (normalized_lookup[candidate] for candidate in candidates if candidate in normalized_lookup),
            None,
        )
        if source_column is None:
            missing.append(canonical_name)
        else:
            rename_map[source_column] = canonical_name

    if missing:
        raise ValueError(f"Raw Kaggle CSV is missing required columns: {', '.join(missing)}")

    df = raw_df.rename(columns=rename_map)[list(rename_map.values())].copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["customer_id"] = df["customer_id"].fillna("UNKNOWN").astype(str).str.strip()
    df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()

    return df.dropna(subset=["invoice_date"])


def _build_enriched_sales(
    normalized_df: pd.DataFrame,
    max_output_rows: int,
    target_start: date,
    target_end: date,
) -> pd.DataFrame:
    df = normalized_df.copy()
    df = df[df["stock_code"].ne("") & df["description"].ne("")]
    df = df.sort_values("invoice_date").reset_index(drop=True)
    if max_output_rows > 0 and len(df) > max_output_rows:
        indexes = _evenly_spaced_indexes(len(df), max_output_rows)
        df = df.iloc[indexes].copy().reset_index(drop=True)

    df["saleDate"] = _rebase_dates(df["invoice_date"], target_start, target_end)
    df["product"] = df.apply(_assign_product, axis=1)
    customer_volume = _customer_positive_volume(df)
    df["segment"] = df.apply(
        lambda row: _assign_segment(row, customer_volume),
        axis=1,
    )
    df["sourceStatus"] = df.apply(_source_status, axis=1)
    df["confirmedOrder"] = df["sourceStatus"].eq("CONFIRMED")
    df["quantity"] = df.apply(_industrial_quantity, axis=1)
    df["amount"] = df.apply(_industrial_amount, axis=1).round(2)
    df.loc[df["sourceStatus"].eq("CANCELLED"), "amount"] = 0.0

    df["productId"] = df["product"].map(lambda product: product.id)
    df["productSku"] = df["product"].map(lambda product: product.sku)
    df["productName"] = df["product"].map(lambda product: product.name)
    df["clientSegmentId"] = df["segment"].map(lambda segment: segment.id)
    df["clientSegmentName"] = df["segment"].map(lambda segment: segment.name)
    df["clientSegmentType"] = df["segment"].map(lambda segment: segment.type)

    return _aggregate_business_sales(df)


def _products_dataframe(company_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": product.id,
                "name": product.name,
                "sku": product.sku,
                "description": product.description,
                "status": "ACTIVE",
                "company_id": company_id,
            }
            for product in PRODUCTS
        ]
    )


def _segments_dataframe(company_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": segment.id,
                "name": segment.name,
                "type": segment.type,
                "description": segment.description,
                "active": True,
                "company_id": company_id,
            }
            for segment in SEGMENTS
        ]
    )


def _sales_dataframe(enriched_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sale_date": enriched_df["saleDate"],
            "quantity": enriched_df["quantity"],
            "amount": enriched_df["amount"],
            "confirmed_order": enriched_df["confirmedOrder"],
            "source_status": enriched_df["sourceStatus"],
            "product_id": enriched_df["productId"],
            "client_segment_id": enriched_df["clientSegmentId"],
        }
    )


def _sales_export_dataframe(enriched_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({column: enriched_df[column] for column in SALES_EXPORT_COLUMNS})


def _aggregate_business_sales(enriched_df: pd.DataFrame) -> pd.DataFrame:
    grouping_columns = [
        "saleDate",
        "productId",
        "productSku",
        "productName",
        "clientSegmentId",
        "clientSegmentName",
        "clientSegmentType",
        "sourceStatus",
        "confirmedOrder",
    ]
    aggregated_df = (
        enriched_df.groupby(grouping_columns, as_index=False)
        .agg({"quantity": "sum", "amount": "sum"})
        .sort_values(grouping_columns)
        .reset_index(drop=True)
    )
    aggregated_df["quantity"] = aggregated_df["quantity"].astype(int).clip(lower=0)
    aggregated_df["amount"] = aggregated_df["amount"].round(2).clip(lower=0)
    return aggregated_df


def _assign_product(row: pd.Series) -> ProductDefinition:
    key = f"{row['stock_code']}|{row['description']}"
    return PRODUCTS[_stable_int(key) % len(PRODUCTS)]


def _assign_segment(
    row: pd.Series,
    customer_volume: dict[str, float],
) -> SegmentDefinition:
    customer_id = str(row["customer_id"])
    volume = customer_volume.get(customer_id, 0.0)
    high_volume_threshold = _safe_quantile(list(customer_volume.values()), 0.75)
    mid_volume_threshold = _safe_quantile(list(customer_volume.values()), 0.45)
    country = str(row["country"]).upper()

    if volume >= high_volume_threshold:
        return SEGMENTS[0]

    if country not in {"UNITED KINGDOM", "UNKNOWN"} and volume >= mid_volume_threshold:
        return SEGMENTS[2]

    if _stable_float(customer_id + country) > 0.72:
        return SEGMENTS[2]

    return SEGMENTS[1]


def _source_status(row: pd.Series) -> str:
    invoice = str(row["invoice"]).upper()
    score = _stable_float(f"{row['invoice']}|{row['stock_code']}|{row['customer_id']}")
    if invoice.startswith("C") or row["quantity"] < 0 or row["unit_price"] < 0:
        return "CANCELLED" if score < 0.45 else "PENDING"

    if score < 0.002:
        return "CANCELLED"
    if score < 0.040:
        return "PENDING"
    return "CONFIRMED"


def _industrial_quantity(row: pd.Series) -> int:
    product: ProductDefinition = row["product"]
    raw_quantity = max(abs(float(row["quantity"])), 1.0)
    segment: SegmentDefinition = row["segment"]
    segment_quantity_factor = 1.18 if segment.type == "GRAND_COMPTE" else 1.0
    adjusted = round(raw_quantity * product.quantity_scale * segment_quantity_factor)
    return max(1, min(product.max_quantity, int(adjusted)))


def _industrial_amount(row: pd.Series) -> float:
    if row["sourceStatus"] == "CANCELLED":
        return 0.0

    product: ProductDefinition = row["product"]
    segment: SegmentDefinition = row["segment"]
    signal_factor = 0.85 + 0.30 * _stable_float(
        f"{row['stock_code']}|{row['customer_id']}"
    )
    sale_date = date.fromisoformat(row["saleDate"])
    year_factor = 1 + 0.055 * max(sale_date.year - 2021, 0)
    seasonal_factor = SEASONAL_FACTORS[sale_date.month]
    jitter = 0.97 + _stable_float(f"{row['invoice']}|{row['stock_code']}") * 0.06

    return (
        row["quantity"]
        * product.unit_price_mad
        * segment.multiplier
        * signal_factor
        * year_factor
        * seasonal_factor
        * jitter
    )


def _customer_positive_volume(df: pd.DataFrame) -> dict[str, float]:
    positive_df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)].copy()
    positive_df["raw_revenue"] = positive_df["quantity"] * positive_df["unit_price"]
    return positive_df.groupby("customer_id")["raw_revenue"].sum().to_dict()


def _rebase_dates(
    dates: pd.Series,
    target_start: date,
    target_end: date,
) -> pd.Series:
    source_min = dates.min()
    source_max = dates.max()
    target_start_ts = pd.Timestamp(target_start)
    target_span_days = max((pd.Timestamp(target_end) - target_start_ts).days, 1)
    source_span_seconds = max((source_max - source_min).total_seconds(), 1)
    offsets = (dates - source_min).dt.total_seconds() / source_span_seconds
    target_dates = target_start_ts + pd.to_timedelta(offsets * target_span_days, unit="D")
    return target_dates.dt.date.map(lambda value: value.isoformat())


def _dataset_summary(sales_export_df: pd.DataFrame) -> dict:
    status_counts = sales_export_df["sourceStatus"].value_counts().to_dict()
    row_count = len(sales_export_df)
    status_percentages = {
        status: round(count / row_count * 100, 2) if row_count else 0
        for status, count in status_counts.items()
    }
    group_sizes = sales_export_df.groupby(["productSku", "clientSegmentType"]).size()

    return {
        "rowCount": row_count,
        "dateMin": str(sales_export_df["saleDate"].min()) if row_count else None,
        "dateMax": str(sales_export_df["saleDate"].max()) if row_count else None,
        "productCount": int(sales_export_df["productSku"].nunique()),
        "segmentCount": int(sales_export_df["clientSegmentType"].nunique()),
        "amountMin": round(float(sales_export_df["amount"].min()), 2) if row_count else None,
        "amountMax": round(float(sales_export_df["amount"].max()), 2) if row_count else None,
        "amountAverage": round(float(sales_export_df["amount"].mean()), 2) if row_count else None,
        "amountTotal": round(float(sales_export_df["amount"].sum()), 2) if row_count else None,
        "sourceStatusCounts": status_counts,
        "sourceStatusPercentages": status_percentages,
        "minRowsPerProductSegment": int(group_sizes.min()) if not group_sizes.empty else 0,
        "maxRowsPerProductSegment": int(group_sizes.max()) if not group_sizes.empty else 0,
    }


def _load_sql_template(paths: dict[str, Path]) -> str:
    return f"""-- Previzma B2B generated dataset load helper
-- Option A: Supabase Table Editor CSV import
--   1. Import {paths['products'].as_posix()} into products
--   2. Import {paths['client_segments'].as_posix()} into client_segments
--   3. Import {paths['sales'].as_posix()} into sales
--
-- Option B: psql from the project root. Supabase SQL Editor does not support \\copy.

\\copy products(id, name, sku, description, status, company_id) from '{paths['products'].as_posix()}' csv header;
\\copy client_segments(id, name, type, description, active, company_id) from '{paths['client_segments'].as_posix()}' csv header;
\\copy sales(sale_date, quantity, amount, confirmed_order, source_status, product_id, client_segment_id) from '{paths['sales'].as_posix()}' csv header;

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
"""


def _normalize_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _stable_float(value: str) -> float:
    return (_stable_int(value) % 10_000) / 10_000


def _safe_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0

    return float(pd.Series(values).quantile(quantile))


def _evenly_spaced_indexes(row_count: int, limit: int) -> list[int]:
    if row_count <= limit:
        return list(range(row_count))

    if limit <= 1:
        return [row_count - 1]

    return sorted(
        {
            round(index * (row_count - 1) / (limit - 1))
            for index in range(limit)
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
