CREATE TABLE cs230_finance_data.vixy_stocks WITH (
    format = 'PARQUET',
    external_location = 's3://cs230-market-data-2025/vixy_stocks'
) AS
SELECT
    stock_symbol,
    CAST(date AS DATE) as date,
    CAST(open AS double) as open,
    CAST(high AS double) as high,
    CAST(low AS double) as low,
    CAST(close AS double) as close,
    CAST(adj_close AS double) as adj_close,
    CAST(volume AS double) as volume,
    (
        LEAD(CAST(close AS double), 1) OVER (
            ORDER BY
                date
        ) - LAG(CAST(close AS double), 1) OVER (
            ORDER BY
                date
        )
    ) AS close_lead_lag_diff
FROM
    "cs230_finance_data"."fnspid_stock_prices_w_path_symbol_t"
where
    stock_symbol = 'vixy'
order by
    date