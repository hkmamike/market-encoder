CREATE TABLE cs230_finance_data.paired_vixy_w_titles_v3 WITH (
    format = 'PARQUET',
    external_location = 's3://cs230-market-data-2025/paired_vixy_w_titles_v3'
) AS -- Step 1: Assign a random row number to every row
WITH RandomizedRows AS (
    SELECT
        *,
        -- Assign a unique row number in a random order
        ROW_NUMBER() OVER (
            ORDER BY
                RANDOM()
        ) AS rn
    FROM
        titles_10_to_vixy_open_lead_close_lag -- 3. THIS IS YOUR TABLE FROM THE EXAMPLE
    WHERE
        date IS NOT NULL
) -- Step 2: Join the table to itself using a "sliding window" (t1.rn = t2.rn - 1)
-- This pairs each row with the *next* row in the random list.
SELECT
    t1.aggregated_article_title AS concatarticles1,
    t1.date AS date1,
    t1.open_lead_close_lag_change_pct AS vol_diff1,
    t2.aggregated_article_title AS concatarticles2,
    t2.date AS date2,
    t2.open_lead_close_lag_change_pct AS vol_diff2,

    t2.open_lead_close_lag_change_pct - t1.open_lead_close_lag_change_pct AS vol_1_vs_2
FROM
    RandomizedRows t1 -- Join each row (t1) to the immediately following row (t2)
    JOIN RandomizedRows t2 ON t1.rn = t2.rn - 1