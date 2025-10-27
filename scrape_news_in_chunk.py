"""
Scrapes financial news articles from URLs provided in a Hugging Face dataset
and uploads the results directly to an S3 bucket in chunked CSV files.

This script is designed to handle very large datasets by processing them
in memory-efficient chunks and supports processing specific row ranges.

This script performs the following steps:
1. Loads the dataset as a stream from Hugging Face.
2. Iterates through the stream, processing articles in chunks (e.g., 10,000 rows),
   respecting the start and end row boundaries.
3. For each chunk:
    a. Scrapes the text content from the URLs.
    b. Adds the scraped text as a new column named 'scraped_text'.
    c. Saves the processed chunk as a separate CSV (e.g., 'part-00001.csv').
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
  python scrape_news_in_chunk.py

  # To process ONLY the first 1%, 157,000 rows
  python scrape_news_in_chunk.py --end-row 157000 --s3-prefix 1-percent

  # To process the 2nd percent, 157,000 rows (starting from row 157,000)
  python scrape_news_in_chunk.py --start-row 157000 --end-row 314000 --s3-prefix 1-2-percent

  # To process the REMAINING rows after 1st percent (starting from row 157,000)
  python scrape_news_in_chunk.py --start-row 157000 --s3-prefix 1-99-percent

  # To upload the first 1% 157,000 rows WITHOUT scraping
  python scrape_news_in_chunk.py --end-row 157000 --no-scrape --s3-prefix 1-percent-no-scrape

  # To upload the all data WITHOUT scraping
  python scrape_news_in_chunk.py --end-row 15700000 --no-scrape --s3-prefix all-data-no-scrape
"""

import pandas as pd
from datasets import load_dataset, Features, Value
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse
import newspaper
from io import StringIO
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# --- Configuration ---
DATASET_PATH = "Zihan1004/FNSPID"
REQUEST_TIMEOUT = 10
USER_AGENT = 'Mozilla/5.0'
HUGGING_FACE_SPLIT = 'train' # train/validation/test

# --- S3 Configuration ---
S3_BUCKET_NAME = "cs230-market-data-2025"
AWS_REGION = "us-east-1"
AWS_PROFILE_NAME = "team-s3-uploader"

# --- Chunking Configuration ---
CHUNK_SIZE = 15700000  # Process 10,000 articles at a time

def scrape_text_v1(url):
    """
    Scrapes the main article text from a URL using the newspaper3k library.
    """
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        return article.text
    except newspaper.article.ArticleException as e:
        print(f"Error processing article at {url}: {e}")
        return ""
    except Exception as e:
        print(f"A general error occurred for {url}: {e}")
        return ""

def upload_df_to_s3(df, bucket, s3_object_key, region):
    """
    Uploads a pandas DataFrame to an S3 bucket as a CSV file.
    """
    print(f"Uploading DataFrame to bucket '{bucket}' as '{s3_object_key}'...")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=region)
    s3_client = session.client('s3')
    try:
        s3_client.put_object(Bucket=bucket, Key=s3_object_key, Body=csv_content)
        print("Upload Successful!")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        print("Error: AWS credentials not found.")
        return False
    except ClientError as e:
        print(f"An S3 client error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def process_and_upload_chunk(chunk_list, chunk_index, s3_prefix, s3_bucket, region, no_scrape=False):
    """
    Converts a list of rows to a DataFrame, scrapes, and uploads to S3.
    """
    if not chunk_list:
        print("Empty chunk, skipping.")
        return

    print(f"--- Processing chunk {chunk_index} ({len(chunk_list)} rows) ---")
    
    # 1. Convert chunk to DataFrame
    df = pd.DataFrame(chunk_list)
    
    # 2. Scrape the text, unless --no-scrape is specified
    if not no_scrape:
        tqdm.pandas(desc=f"Scraping Chunk {chunk_index}")
        df['scraped_text'] = df['Url'].progress_apply(scrape_text_v1)
    else:
        print(f"Skipping scraping for chunk {chunk_index} as requested.")
    
    # 3. Determine S3 object key for this chunk
    # zfill(5) pads the number (e.g., 1 -> 00001) for correct file ordering
    s3_object_key = f"{s3_prefix.rstrip('/')}/part-{chunk_index:05d}.csv"
    
    # 4. Upload this chunk
    upload_df_to_s3(df, s3_bucket, s3_object_key, region)
    print(f"--- Chunk {chunk_index} completed ---")

def main():
    """
    Main function to load the dataset, scrape articles in chunks,
    and save the updated data to S3.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Scrape financial news articles in chunks from row ranges.")
    
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="processed_data/fnspid_scraped",
        help="The S3 'folder' (prefix) to upload chunked CSV files to."
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="The row index to start processing from (inclusive). Defaults to 0."
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=-1,
        help="The row index to stop processing at (exclusive). Defaults to -1 (no limit)."
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="If specified, skips the scraping process and uploads the raw data directly."
    )
    args = parser.parse_args()

    # --- Config Validation ---
    if S3_BUCKET_NAME == "your-s3-bucket-name-here":
        print("--- CONFIGURATION NEEDED ---")
        print("Please edit the script and replace 'your-s3-bucket-name-here' with your actual S3 bucket name.")
        return

    # --- Schema Definition ---
    feature_schema = Features({
        'Date': Value('string'),
        'Article_title': Value('string'),
        'Stock_symbol': Value('string'),
        'Url': Value('string'),
        'Publisher': Value('string')
    })

    # --- Argument logic for row limits ---
    start_row = args.start_row
    end_row = args.end_row
        
    print(f"--- Starting Scrape ---")
    print(f"Processing rows from {start_row} up to {end_row if end_row != -1 else 'the end'}.")
    print(f"Uploading chunks to: s3://{S3_BUCKET_NAME}/{args.s3_prefix}/")


    # --- Load Dataset (Streaming) ---
    # We always load the full 'train' split now
    print(f"Loading dataset stream from Hugging Face ({DATASET_PATH}, split='{HUGGING_FACE_SPLIT}')...")
    streaming_dataset = load_dataset(
        DATASET_PATH,
        split=HUGGING_FACE_SPLIT,
        features=feature_schema,
        streaming=True
    )

    # --- Chunked Processing Loop ---
    chunk = []
    chunk_index = 0
    total_rows_processed = 0
    current_row_index = 0

    # Use tqdm to create a progress bar for the stream
    stream_progress = tqdm(streaming_dataset, desc="Streaming rows")

    for row in stream_progress:
        # 1. Skip rows before our start_row
        if current_row_index < start_row:
            if (current_row_index + 1) % 100000 == 0: # Update progress bar periodically while skipping
                 stream_progress.set_description(f"Skipping to start row... at {current_row_index + 1}")
            current_row_index += 1
            continue

        # 2. Stop if we've reached our end_row
        if end_row != -1 and current_row_index >= end_row:
            print(f"\nReached end-row limit of {end_row}. Stopping stream.")
            break

        # Add row to the current chunk
        chunk.append(row)

        # Check if the chunk is full
        if len(chunk) >= CHUNK_SIZE:
            # Pass the *absolute* chunk index for unique file naming
            absolute_chunk_index = (current_row_index // CHUNK_SIZE)
            process_and_upload_chunk(
                chunk, absolute_chunk_index, args.s3_prefix, S3_BUCKET_NAME, AWS_REGION, args.no_scrape
            )
            total_rows_processed += len(chunk)
            chunk = []  # Reset the chunk
        
        current_row_index += 1
            
    # Process the final, partial chunk
    if chunk:
        print("Processing the final partial chunk...")
        absolute_chunk_index = (current_row_index // CHUNK_SIZE)
        process_and_upload_chunk(
            chunk, absolute_chunk_index, args.s3_prefix, S3_BUCKET_NAME, AWS_REGION, args.no_scrape
        )
        total_rows_processed += len(chunk)

    print(f"\n--- Scraping complete ---")
    print(f"Total rows processed in this run: {total_rows_processed}")
    print(f"Last row index processed: {current_row_index - 1}")
    print(f"Data is available in S3 at: s3://{S3_BUCKET_NAME}/{args.s3_prefix}/")


if __name__ == '__main__':
    main()