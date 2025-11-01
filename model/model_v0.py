# This script queries Athena from a local Python environment.
#
# Read README and configure the aws credential, follow the instructions: "aws configure ..."
# to run: python model_v0.py

import pandas as pd
import awswrangler as wr
import sys
import boto3  # <-- Import boto3

print("--- Step 1: Libraries imported ---")

# --- IMPORTANT: Set these variables before running ---
AWS_REGION = 'us-east-2'
S3_STAGING_DIR = 's3://cs230-market-data-2025/athena-query-results/'
ATHENA_DB = 'cs230_finance_data'
SQL_QUERY = "SELECT * FROM trainning_v0_example LIMIT 100"
AWS_PROFILE_NAME = "team-s3-uploader"
# ----------------------------------------------------

print(f"--- Step 2: Configuration set for {ATHENA_DB} ---")
print(f"--- Step 3: Querying Data ---")
print(f"Querying data from {ATHENA_DB}.trainning_v0_example...")

# Define df in a wider scope
df = None 

try:
    # Create a boto3 session with the specified region
    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=AWS_REGION)

    # Run the query and load results into a Pandas DataFrame
    # No boto3_session is needed. awswrangler will automatically
    # find the credentials you set up with 'aws configure'.
    df = wr.athena.read_sql_query(
        sql=SQL_QUERY,
        database=ATHENA_DB,
        s3_output=S3_STAGING_DIR,
        boto3_session=session,
    )

    print("\nQuery successful! Data loaded into DataFrame.")
    
    # Display the first 5 rows
    print(df.head())

except Exception as e:
    print(f"\nAn error occurred:")
    print(e)


# -----------------
# Use Data
# -----------------
print(f"\n--- Step 4: Using Your Data ---")

# Check if the DataFrame 'df' was successfully created in Step 3
if df is not None:
    # Example: Show data types
    print("\nDataFrame Info:")
    df.info()

    # Example: Get statistics on the 'SameBucket' column
    print("\nValue Counts for 'samebucket' column:")
    print(df['samebucket'].value_counts())
else:
    print("\nDataFrame was not created (likely due to an error in Step 3).")
