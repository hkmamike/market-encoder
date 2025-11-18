CREATE TABLE cs230_finance_data.paired_spy_w_titles_dedupped WITH (
    format = 'PARQUET',
    external_location = 's3://cs230-market-data-2025/paired_spy_w_titles_dedupped'
) AS 
WITH 
Stock_Metrics_Raw AS (
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
            LEAD(CAST(volume AS double), 1) OVER (ORDER BY date) 
            - LAG(CAST(volume AS double), 1) OVER (ORDER BY date)
        ) AS volume_delta,
        (
            (
                LEAD(CAST(volume AS double), 1) OVER (ORDER BY date) 
                - LAG(CAST(volume AS double), 1) OVER (ORDER BY date)
            ) / NULLIF(LAG(CAST(volume AS double), 1) OVER (ORDER BY date), 0)
        ) AS volume_change_ratio,
        (
            LEAD(CAST(open AS double), 1) OVER (ORDER BY date) 
            - LAG(CAST(close AS double), 1) OVER (ORDER BY date)
        ) AS open_lead_close_lag_diff,
        (
            (
                LEAD(CAST(open AS double), 1) OVER (ORDER BY date) 
                - LAG(CAST(close AS double), 1) OVER (ORDER BY date)
            ) * 100.0 / NULLIF(LAG(CAST(close AS double), 1) OVER (ORDER BY date), 0)
        ) AS open_lead_close_lag_change_pct
    FROM
        "cs230_finance_data"."fnspid_stock_prices_w_path_symbol_t"
    WHERE
        stock_symbol = 'spy'
),
Stock_Metrics AS (
    SELECT
        *,
        NTILE(10) OVER (
            ORDER BY
                volume_delta
        ) AS volume_delta_decile_bucket,
        NTILE(10) OVER (
            ORDER BY
                volume_change_ratio
        ) AS volume_change_ratio_decile_bucket
    FROM
        Stock_Metrics_Raw
    WHERE
        volume_delta IS NOT NULL
        AND volume_change_ratio IS NOT NULL
),
Article_Groups AS (
    SELECT
        clean_date,
        article_title,
        -- Create a group ID for every 10 articles
        (rn - 1) / 10 as article_group
    FROM
        (
            SELECT DISTINCT
                TRY_CAST(SUBSTR(t2.date, 1, 10) AS DATE) AS clean_date,
                t2.article_title,
                ROW_NUMBER() OVER (
                    PARTITION BY TRY_CAST(SUBSTR(t2.date, 1, 10) AS DATE)
                    ORDER BY RANDOM()
                ) AS rn
            FROM
                cs230_finance_data.articles_no_scrape t2
        ) sub
),
Aggregated_Batches AS (
    SELECT
        t1.date,
        ANY_VALUE(t1.open_lead_close_lag_change_pct) as open_lead_close_lag_change_pct,
        ANY_VALUE(t1.volume_delta_decile_bucket) as volume_delta_decile_bucket,
        ANY_VALUE(t1.volume_change_ratio_decile_bucket) as volume_change_ratio_decile_bucket,
        ARRAY_JOIN(ARRAY_AGG(t2.article_title), '|') as aggregated_article_title
    FROM
        Article_Groups t2
    LEFT JOIN 
        Stock_Metrics t1 ON t2.clean_date = t1.date
    WHERE 
        t1.date IS NOT NULL
    GROUP BY
        t1.date,
        t2.article_group
),
Randomized_Rows AS (
    SELECT
        *,
        -- Assign a unique row number in a random order
        ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
    FROM
        Aggregated_Batches
)
SELECT
    t1.aggregated_article_title AS concatarticles1,
    t1.date AS date1,
    t1.volume_delta_decile_bucket AS bucket_delta1,
    t1.volume_change_ratio_decile_bucket AS bucket_ratio1,
    -- t1.open_lead_close_lag_change_pct AS vol_diff1,
    
    t2.aggregated_article_title AS concatarticles2,
    t2.date AS date2,
    t2.volume_delta_decile_bucket AS bucket_delta2,
    t2.volume_change_ratio_decile_bucket AS bucket_ratio2,
    -- t2.open_lead_close_lag_change_pct AS vol_diff2,

    CASE
        WHEN t1.volume_delta_decile_bucket = t2.volume_delta_decile_bucket THEN 1
        ELSE 0
    END AS samebucket_delta,
    CASE
        WHEN t1.volume_change_ratio_decile_bucket = t2.volume_change_ratio_decile_bucket THEN 1
        ELSE 0
    END AS samebucket_ratio
    -- t2.open_lead_close_lag_change_pct - t1.open_lead_close_lag_change_pct AS vol_1_vs_2
FROM
    Randomized_Rows t1
JOIN 
    Randomized_Rows t2 ON t1.rn = t2.rn - 1;