import awswrangler as wr
import pandas as pd

# Define your AWS Glue/Athena database and the table.
# Replace 'your_database_name' with the actual database name where
# cs230_finance_data.articles_no_scrape resides (e.g., 'cs230_finance_data').
DATABASE_NAME = 'cs230_finance_data'
TABLE_NAME = 'articles_no_scrape'

# Your S3 staging directory, which Athena needs to write query results.
# The URL you provided looks like an Athena query result path, which is where
# Athena writes the result files. You typically don't query this S3 path directly,
# but rather use the database and table names. The S3 output is needed for the wr.athena.read_sql_query call.
# Use your designated Athena query result bucket:
S3_OUTPUT_PATH = 's3://cs230-market-data-2025/athena-query-results/' 

QUERY = f"SELECT X1, X2, Y FROM {DATABASE_NAME}.{TABLE_NAME}" # Adjust to your desired query

print("Querying Athena...")

# Load the data directly into a Pandas DataFrame
try:
    df = wr.athena.read_sql_query(
        sql=QUERY,
        database=DATABASE_NAME,
        s3_output=S3_OUTPUT_PATH,
        # Consider ctas_approach=True (default) for faster results on large datasets, 
        # but it requires Glue permissions to create/delete temporary tables.
    )
    print(f"Data loaded successfully. DataFrame shape: {df.shape}")
except Exception as e:
    print(f"Error querying Athena: {e}")
    # Handle error or exit