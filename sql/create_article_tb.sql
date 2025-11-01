CREATE EXTERNAL TABLE IF NOT EXISTS cs230_finance_data.articles_no_scrape (
    Date STRING,
    Article_title STRING,
    Stock_symbol STRING,
    Url STRING,
    Publisher STRING
) -- Using OpenCSVSerde as specified in your successful example.
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"') STORED AS TEXTFILE -- This LOCATION points to the *directory* containing all your part-files.
LOCATION 's3://cs230-market-data-2025/all-data-no-scrape/' -- Using the exact TBLPROPERTIES from your working example.
TBLPROPERTIES (
    'has_header' = 'true',
    'skip.header.line.count' = '1'
) -- why