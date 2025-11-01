CREATE TABLE cs230_finance_data.volume_change_bucket_match_apple WITH (
    format = 'PARQUET',
    external_location = 's3://cs230-market-data-2025/volume_change_bucket_match_apple'
) AS -- Step 1: Assign a random row number to every row
WITH RandomizedRows AS (
    SELECT
        date,
        aggregated_article_title,
        volume_delta_decile_bucket,
        volume_change_ratio_decile_bucket,
        -- Assign a unique row number in a random order
        ROW_NUMBER() OVER (
            ORDER BY
                RANDOM()
        ) AS rn
    FROM
        titles_10_to_apple_buckets -- 3. THIS IS YOUR TABLE FROM THE EXAMPLE
    WHERE
        date IS NOT NULL
) -- Step 2: Join the table to itself using a "sliding window" (t1.rn = t2.rn - 1)
-- This pairs each row with the *next* row in the random list.
SELECT
    t1.aggregated_article_title AS concatarticles1,
    t1.date AS date1,
    t2.aggregated_article_title AS concatarticles2,
    t2.date AS date2,
    -- Check if they are in the same delta bucket (1 = yes, 0 = no)
    -- CASE
    --     WHEN t1.volume_delta_decile_bucket = t2.volume_delta_decile_bucket
    --     THEN 1
    --     ELSE 0
    -- END AS 	samebucket
    -- ,
    -- Check if they are in the same ratio bucket (1 = yes, 0 = no)
    CASE
        WHEN t1.volume_change_ratio_decile_bucket = t2.volume_change_ratio_decile_bucket THEN 1
        ELSE 0
    END AS samebucket
FROM
    RandomizedRows t1 -- Join each row (t1) to the immediately following row (t2)
    JOIN RandomizedRows t2 ON t1.rn = t2.rn - 1