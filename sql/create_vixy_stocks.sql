CREATE TABLE IF NOT EXISTS cs230_finance_data.vixy_stocks_open_lead_close_lag AS AS
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
        LEAD(CAST(open AS double), 1) OVER (
            ORDER BY
                date
        ) - LAG(CAST(close AS double), 1) OVER (
            ORDER BY
                date
        )
    ) AS open_lead_close_lag_diff,
    -- Calculate volume_change_ratio
    (
        (
            LEAD(CAST(open AS double), 1) OVER (
                ORDER BY
                    date
            ) - LAG(CAST(close AS double), 1) OVER (
                ORDER BY
                    date
            )
        ) * 100.0 / NULLIF(
            LAG(CAST(close AS double), 1) OVER (
                ORDER BY
                    date
            ),
            0
        )
    ) AS open_lead_close_lag_change_pct

FROM
    "cs230_finance_data"."fnspid_stock_prices_w_path_symbol_t"
where
    stock_symbol = 'vixy'
order by
    date