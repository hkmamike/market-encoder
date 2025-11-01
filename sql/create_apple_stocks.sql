CREATE TABLE cs230_finance_data.apple_stocks WITH (
    format = 'PARQUET',
    external_location = 's3://cs230-market-data-2025/apple_stocks'
) AS
SELECT
    stock_symbol,
    CAST(date AS DATE) as date,
    CAST(open AS double) as open,
    CAST(high AS double) as high,
    CAST(low AS double) as low,
    CAST(close AS double) as close,
    CAST(adj_close AS double) as adj_close,
    CAST(volume AS double) as volume
FROM
    "cs230_finance_data"."fnspid_stock_prices_w_path_symbol_t"
where
    stock_symbol = 'aapl'
order by
    date