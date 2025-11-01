CREATE TABLE cs230_finance_data.apple_stock_bucketized WITH (
    format = 'PARQUET',
    external_location = 's3://cs230-market-data-2025/apple_stock_bucketized'
) AS WITH calculated_changes AS (
    SELECT
        stock_symbol,
        date,
        open,
        high,
        low,
        close,
        adj_close,
        volume,
        -- Calculate volume_delta
        (
            LEAD(volume, 1) OVER (
                PARTITION BY stock_symbol
                ORDER BY
                    date
            ) - LAG(volume, 1) OVER (
                PARTITION BY stock_symbol
                ORDER BY
                    date
            )
        ) AS volume_delta,
        -- Calculate volume_change_ratio
        (
            (
                LEAD(volume, 1) OVER (
                    PARTITION BY stock_symbol
                    ORDER BY
                        date
                ) - LAG(volume, 1) OVER (
                    PARTITION BY stock_symbol
                    ORDER BY
                        date
                )
            ) / NULLIF(
                LAG(volume, 1) OVER (
                    PARTITION BY stock_symbol
                    ORDER BY
                        date
                ),
                0
            )
        ) AS volume_change_ratio
    FROM
        apple_stocks
) -- Step 2: Select from the CTE and assign decile buckets using NTILE
SELECT
    stock_symbol,
    date,
    open,
    high,
    low,
    close,
    adj_close,
    volume,
    volume_delta,
    volume_change_ratio,
    -- Create decile bucket for volume_delta
    -- NTILE(10) divides the ordered rows into 10 equal buckets
    NTILE(10) OVER (
        PARTITION BY stock_symbol
        ORDER BY
            volume_delta
    ) AS volume_delta_decile_bucket,
    -- Create decile bucket for volume_change_ratio
    NTILE(10) OVER (
        PARTITION BY stock_symbol
        ORDER BY
            volume_change_ratio
    ) AS volume_change_ratio_decile_bucket
FROM
    calculated_changes
WHERE
    volume_delta IS NOT NULL
    AND volume_change_ratio IS NOT NULL
ORDER BY
    stock_symbol,
    date