CREATE TABLE cs230_finance_data.titles_10_to_vixy WITH (
    format = 'PARQUET',
    external_location = 's3://cs230-market-data-2025/titles_10_to_vixy'
) AS WITH T2_ARTICLES AS (
    SELECT
        *,
        -- Selects all columns from sub query
        -- Create a group ID for every 10 articles
        (rn - 1) / 10 as article_group
    FROM
        (
            SELECT
                TRY_CAST(SUBSTR(t2.date, 1, 10) AS DATE) AS clean_date,
                t2.*,
                -- Pass through all original columns
                -- Add row number partitioned by the clean_date
                ROW_NUMBER() OVER (
                    PARTITION BY TRY_CAST(SUBSTR(t2.date, 1, 10) AS DATE)
                    ORDER BY
                        t2.article_title -- using title as an arbitrary but stable sort
                ) as rn
            FROM
                cs230_finance_data.articles_no_scrape t2
        ) sub
) -- T1_PRICES CTE has been removed as requested.
-- This query is now aggregated based on your new requirements
SELECT
    t1.date,
    -- Grouping key
    -- Use ANY_VALUE for t1 columns since they are all the same for a given date
    ANY_VALUE(t1.stock_symbol) as label_stock_symbol,
    ANY_VALUE(t1.open) as open,
    ANY_VALUE(t1.high) as high,
    ANY_VALUE(t1.low) as low,
    ANY_VALUE(t1.close) as close,
    ANY_VALUE(t1.adj_close) as adj_close,
    ANY_VALUE(t1.volume) as volume,
    ANY_VALUE(t1.close_lead_lag_diff) as close_lead_lag_diff,
    -- Aggregate article titles with a '|' separator
    -- Use ARRAY_JOIN(ARRAY_AGG(...)) as the alternative to STRING_AGG
    ARRAY_JOIN(ARRAY_AGG(t2.article_title), '|') as aggregated_article_title -- Other t2 columns (article_stock_symbol, url, publisher) are dropped
    -- as requested by grouping
FROM
    T2_ARTICLES t2
    LEFT JOIN -- Join directly to the table instead of using the CTE
    cs230_finance_data.vixy_stocks t1 ON t2.clean_date = t1.date -- Group by the date from the prices table AND the new article group
GROUP BY
    t1.date,
    t2.article_group