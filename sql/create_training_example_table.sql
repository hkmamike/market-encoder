/*
-- This is an Athena CREATE TABLE AS (CTAS) query.
-- It creates a new table and writes new data to S3.
*/

CREATE TABLE trainning_v0_example
WITH (
    -- The S3 location where the new data files will be written.
    external_location = 's3://cs230-market-data-2025/training/',
    
    -- We'll use TEXTFILE (CSV-like) with a comma delimiter.
    -- You could also use 'PARQUET' or 'JSON' for better performance.
    format = 'TEXTFILE',
    field_delimiter = ','
)
AS
-- This SELECT statement generates 100 rows of fake data
SELECT
    CONCAT('Text', CAST(n AS VARCHAR), '+', CAST(n + 100 AS VARCHAR)) AS ConcatArticles1,
    CONCAT('Text', CAST(n + 200 AS VARCHAR), '+', CAST(n + 300 AS VARCHAR)) AS ConcatArticles2,
    CAST(n % 2 AS VARCHAR) AS SameBucket,
    CAST(current_date - (n DAY) AS VARCHAR) AS Date1,
    CAST(current_date - ((n + 1) DAY) AS VARCHAR) AS Date2
FROM
    -- This is a trick to generate a sequence of numbers from 1 to 100
    (SELECT 1) AS t_dummy(dummy_col)
    CROSS JOIN
    UNNEST(SEQUENCE(1, 100)) AS t_seq(n);