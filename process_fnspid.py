"""
Loads data from a Hugging Face dataset and uploads the results directly
to an S3 bucket in chunked CSV files.

This script is designed to handle very large datasets by processing them
in memory-efficient chunks and supports processing specific row ranges.

This script performs the following steps:
1. Loads the dataset as a stream from Hugging Face.
2. Iterates through the stream, processing articles in chunks (e.g., 10,000 rows),
   respecting the start and end row boundaries.
3. For each chunk:
    a. Trims and cleans whitespace
4. Uploads each CSV chunk directly to a specified S3 prefix (folder).

Valid Command-Line Arguments:
  --start-row [INT]    : (Optional) The row index to start processing from (inclusive).
                           Defaults to 0.
  --end-row [INT]      : (Optional) The row index to stop processing at (exclusive).
                           Defaults to -1 (no limit).
  --s3-prefix [STRING] : (Optional) The S3 'folder' (prefix) to upload the CSV chunks to.
                           (e.g., 'data/fnspid_scraped').
  --no-scrape          : (Optional) If provided, skips the scraping process and uploads
                           the raw data directly to S3.
                           Defaults to 'processed_data/fnspid_scraped'.

Example Usage:
  # To process the full 'train' dataset (default)
  python process_fnspid.py

  # To process ONLY the first 1%, 157,000 rows
  python process_fnspid.py --end-row 157000 --s3-prefix 1-percent

  # To process the 2nd percent, 157,000 rows (starting from row 157,000)
  python process_fnspid.py --start-row 157000 --end-row 314000 --s3-prefix 1-2-percent

  # To process the REMAINING rows after 1st percent (starting from row 157,000)
  python process_fnspid.py --start-row 157000 --s3-prefix 1-99-percent

  # To upload the first 1% 157,000 rows WITHOUT scraping
  python process_fnspid.py --end-row 157000 --no-scrape --s3-prefix 1-percent-no-scrape

  # To upload the all data WITHOUT scraping
  python process_fnspid.py --end-row 15700000 --no-scrape --s3-prefix all-data-no-scrape
"""

import pandas as pd
from datasets import load_dataset, Features, Value
from tqdm import tqdm
import argparse
import boto3
import csv
from boto3.s3.transfer import TransferConfig
import uuid
import os

# --- Configuration ---
DATASET_PATH = "sabareesh88/FNSPID_nasdaq"
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0"
HUGGING_FACE_SPLIT = "train"  # train/validation/test

# --- S3 Configuration ---
S3_BUCKET_NAME = "cs230-market-data-2025"
AWS_REGION = "us-east-2"
AWS_PROFILE_NAME = "team-s3-uploader"

# --- Chunking Configuration ---
CHUNK_SIZE = 100000  # Process 100,000 articles at a time
# Define a standard multi-part chunk size (e.g., 50MB) for the transfer manager
MB_50 = 50 * 1024 * 1024
LOCAL_TEMP_DIR = "temp_s3_uploads"

TARGET_DTYPES = {
    "Date": "string",
    "Article_title": "string",
    "Stock_symbol": "string", 
    "Url": "string",
    "Publisher": "string",
    "Author": "string",
    "Article": "string",
    "Lsa_summary": "string",
    "Luhn_summary": "string",
    "Textrank_summary": "string",
    "Lexrank_summary": "string",
}

TEXT_CLEANUP_COLUMNS = [
    "Article_title",
    "Article",
    "Publisher",
    "Author",
    "Lsa_summary",
    "Luhn_summary",
    "Textrank_summary",
    "Lexrank_summary",
]


def upload_df_to_s3(df, bucket, s3_object_key, region, chunk_index):
    """
    Uploads a pandas DataFrame to an S3 bucket as a CSV file.
    """
    if not os.path.exists(LOCAL_TEMP_DIR):
        os.makedirs(LOCAL_TEMP_DIR)

    temp_filename = os.path.join(
        LOCAL_TEMP_DIR, f"chunk-{chunk_index}-{uuid.uuid4()}.csv"
    )
    print(f"Saving chunk temporarily to disk: {temp_filename}")
    try:
        # Write the entire CSV content to disk
        df.to_csv(
            temp_filename, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8"
        )

    except Exception as e:
        print(f"❌ Error during local CSV write: {e}")
        return False

    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=region)
    s3_client = session.client("s3")
    config = TransferConfig(multipart_threshold=MB_50, multipart_chunksize=MB_50)

    try:
        print(
            f"Starting multi-part upload of {temp_filename} to S3 as {s3_object_key}..."
        )

        s3_client.upload_file(temp_filename, bucket, s3_object_key, Config=config)
        print("S3 Upload Successful!")

    except Exception as e:
        print(f"❌ An S3 client error occurred during multi-part upload: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"Cleaned up temporary file: {temp_filename}")


def clean_internal_newlines(df):
    """Replaces all internal newline characters with a single space."""
    for col in TEXT_CLEANUP_COLUMNS:
        # Check if the column exists and is a string type
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.replace(r"[\n\r]+", " ", regex=True).str.strip()
            df[col] = df[col].str.replace('"', "'", regex=False)
            df[col] = df[col].str.strip()
    return df


def process_chunk(
    chunk_list, chunk_index, s3_prefix, s3_bucket, region, no_scrape=False
):
    """
    Converts a list of rows to a DataFrame, scrapes, and uploads to S3.
    """
    if not chunk_list:
        print("Empty chunk, skipping.")
        return

    print(f"--- Processing chunk {chunk_index} ({len(chunk_list)} rows) ---")

    df = pd.DataFrame(chunk_list)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df = df.astype({k: v for k, v in TARGET_DTYPES.items() if k in df.columns})
    df = clean_internal_newlines(df)

    s3_object_key = f"{s3_prefix.rstrip('/')}/part-{chunk_index:05d}.csv"
    upload_df_to_s3(df, s3_bucket, s3_object_key, region, chunk_index)
    print(f"--- Chunk {chunk_index} completed ---")


def main():
    """
    Main function to load the dataset, scrape articles in chunks,
    and save the updated data to S3.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Scrape financial news articles in chunks from row ranges."
    )

    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="processed_data/sabareesh88",
        help="The S3 'folder' (prefix) to upload chunked CSV files to.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="The row index to start processing from (inclusive). Defaults to 0.",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=-1,
        help="The row index to stop processing at (exclusive). Defaults to -1 (no limit).",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="If specified, skips the scraping process and uploads the raw data directly.",
    )
    args = parser.parse_args()

    # --- Config Validation ---
    if S3_BUCKET_NAME == "your-s3-bucket-name-here":
        print("--- CONFIGURATION NEEDED ---")
        print(
            "Please edit the script and replace 'your-s3-bucket-name-here' with your actual S3 bucket name."
        )
        return

    # --- Argument logic for row limits ---
    start_row = args.start_row
    end_row = args.end_row

    print(
        f"Processing rows from {start_row} up to {end_row if end_row != -1 else 'the end'}."
    )
    print(f"Uploading chunks to: s3://{S3_BUCKET_NAME}/{args.s3_prefix}/")

    # --- Load Dataset ---
    print(
        f"Loading dataset from Hugging Face ({DATASET_PATH}, split='{HUGGING_FACE_SPLIT}')..."
    )
    full_dataset = load_dataset(
        DATASET_PATH,
        split=HUGGING_FACE_SPLIT,
    )

    total_dataset_size = len(full_dataset)
    print(f"Full dataset size: {total_dataset_size} rows")

    # --- Chunked Processing Loop ---
    chunk = []
    total_rows_processed = 0
    current_row_index = 0

    for current_row_index in tqdm(
        range(start_row, total_dataset_size), desc="Processing rows"
    ):
        row = full_dataset[current_row_index]

        if end_row != -1 and current_row_index >= end_row:
            print(f"\nReached end-row limit of {end_row}. Stopping stream.")
            break

        chunk.append(row)

        if len(chunk) >= CHUNK_SIZE:
            absolute_chunk_index = current_row_index // CHUNK_SIZE
            process_chunk(
                chunk,
                absolute_chunk_index,
                args.s3_prefix,
                S3_BUCKET_NAME,
                AWS_REGION,
                args.no_scrape,
            )

            total_rows_processed += len(chunk)
            chunk = []  # Reset the chunk

        current_row_index += 1

    # Process the final, partial chunk
    if chunk:
        print("Processing the final partial chunk...")
        absolute_chunk_index = current_row_index // CHUNK_SIZE
        process_chunk(
            chunk,
            absolute_chunk_index,
            args.s3_prefix,
            S3_BUCKET_NAME,
            AWS_REGION,
            args.no_scrape,
        )

        total_rows_processed += len(chunk)

    else:
        print("No data chunks were processed or found.")

    print(f"Total rows processed in this run: {total_rows_processed}")
    print(f"Last row index processed: {current_row_index - 1}")
    print(f"Data is available in S3 at: s3://{S3_BUCKET_NAME}/{args.s3_prefix}/")


if __name__ == "__main__":
    main()
