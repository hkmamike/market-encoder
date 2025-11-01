CREATE TABLE cs230_finance_data.fnspid_stock_prices_w_path_symbol_t AS
SELECT
    lower("$path") AS path,
    regexp_extract(lower("$path"), '([^/]+)(?=\.csv$)') AS stock_symbol,
    t.date,
    t.open,
    t.high,
    t.low,
    t.close,
    t.adj_close,
    t.volume
FROM cs230_finance_data.fnspid_stock_prices_base t;