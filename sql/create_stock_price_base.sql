CREATE EXTERNAL TABLE IF NOT EXISTS cs230_finance_data.fnspid_stock_prices_base (
    `date` string,
    `open` string,
    `high` string,
    `low` string,
    `close` string,
    `adj_close` string,
    `volume` string
) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"') LOCATION 's3://cs230-market-data-2025/fnspid/stock-prices/full_history/' TBLPROPERTIES (
    'has_header' = 'true',
    'skip.header.line.count' = '1'
)