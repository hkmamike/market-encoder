"""
Scrapes financial news articles from URLs provided in a Hugging Face dataset.
and uploads the result directly to an S3 bucket.

This script performs the following steps:
1. Loads the dataset from Hugging Face, relying on the library's built-in cache.
2. Scrapes the text content from the URLs in the dataset.
3. Adds the scraped text as a new column named 'scraped_text'.
4. Saves the processed data to a new CSV file.

Valid Command-Line Arguments:
  --num-rows [INTEGER] : (Optional) The number of rows to process. If not provided,
                           the entire dataset will be processed.
  --s3-key [STRING]    : (Optional) The desired S3 object key (e.g., 'data/news.csv').
                           If not provided, a default name will be generated.

Example Usage:
  # To process a sample of 100 articles for testing
  python scrape_news.py --num-rows 100 --s3-key processed_data/financial_news_sample_100.csv

  # To process the entire dataset
  python scrape_news.py --s3-key processed_data/financial_news_full.csv
"""
import pandas as pd
from datasets import load_dataset, Features, Value  # <-- IMPORT FEATURES AND VALUE
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
BASE_OUTPUT_FILENAME = 'financial_news_with_text'
REQUEST_TIMEOUT = 10
USER_AGENT = 'Mozilla/5.0'
HUGGING_FACE_SPLIT = 'train'
HUGGING_FACE_Data_COUNT = 100 # 'train' to get the full set

# --- S3 Configuration ---
S3_BUCKET_NAME = "cs230-market-data-2025"
AWS_REGION = "us-east-1"  # e.g., "us-east-1", "us-west-2"
AWS_PROFILE_NAME = "team-s3-uploader"

# def scrape_text_basic(url):
#     """Scrapes the text content from a given URL."""
#     try:
#         response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={'User-Agent': USER_AGENT})
#         response.raise_for_status()  # Raise an exception for bad status codes
#         soup = BeautifulSoup(response.content, 'html.parser')
        
#         # This is a generic approach; you might need to refine selectors 
#         # based on the common structures of the news websites.
#         paragraphs = soup.find_all('p')
#         article_text = ' '.join([p.get_text() for p in paragraphs])
#         return article_text
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching {url}: {e}")
#         return ""
#     except Exception as e:
#         print(f"Error parsing {url}: {e}")
#         return ""

def scrape_text_v1(url):
    """
    Scrapes the main article text from a URL using the newspaper3k library.
    """
    try:
        # Initialize the Article object
        article = newspaper.Article(url)
        
        # Download the HTML
        article.download()
        
        # Parse the article to find the main content
        article.parse()
        
        # Return the extracted clean text
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

    :param df: The pandas DataFrame to upload.
    :param bucket: The S3 bucket to upload to.
    :param s3_object_key: The desired object key (path/filename) in S3.
    :param region: The AWS region of the bucket.
    :return: True if upload was successful, else False.
    """
    print(f"Uploading DataFrame to bucket '{bucket}' as '{s3_object_key}'...")

    # Convert DataFrame to CSV in-memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Create a session using your specific profile
    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=region)
    s3_client = session.client('s3')
    try:
        s3_client.put_object(Bucket=bucket, Key=s3_object_key, Body=csv_content)
        print("Upload Successful!")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        print("Error: AWS credentials not found.")
        print("Please configure your credentials, for example, in ~/.aws/credentials")
        return False
    except ClientError as e:
        print(f"An S3 client error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    """
    Main function to load the dataset, scrape articles,
    and save the updated data.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Scrape financial news articles from URLs.")
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Number of rows to process from the dataset. Processes all rows by default."
    )
    parser.add_argument(
        "--s3-key",
        type=str,
        default=None,
        help="The S3 object key (path/filename) for the output file."
    )
    args = parser.parse_args()

    # Before running, check if the user has updated the placeholder bucket name
    if S3_BUCKET_NAME == "your-s3-bucket-name-here":
        print("--- CONFIGURATION NEEDED ---")
        print("Please edit 'scrape_news.py' and replace 'your-s3-bucket-name-here' with your actual S3 bucket name.")
        return

    # --- (FIX 1) Define the schema to prevent type-inference errors ---
    # We force all columns to be read as strings
    feature_schema = Features({
        'Date': Value('string'),
        'Article_title': Value('string'),
        'Stock_symbol': Value('string'),
        'Url': Value('string'),
        'Publisher': Value('string')
    })

    # --- Load Dataset (Hugging Face handles caching) ---
    print(f"Loading dataset from Hugging Face ({DATASET_PATH})...")
    print("The 'datasets' library will automatically use a local cache if available.")
    # dataset = load_dataset(DATASET_PATH, split='train')
    streaming_dataset = load_dataset(
        DATASET_PATH,
        split=HUGGING_FACE_SPLIT,
        features=feature_schema,  # <-- PASS THE CORRECT 'features' ARGUMENT
        streaming=True
    )
    partial_data_iterable = streaming_dataset.take(HUGGING_FACE_Data_COUNT)
    print(f"Datastreaming Setup is done.")
    partial_data_list = list(partial_data_iterable)
    print(f"Datastreaming has been cast to list in memoery.")
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(partial_data_list)
    print(f"Loaded {len(df)} articles.")
    
    # --- Limit rows for processing if specified ---
    if args.num_rows is not None and args.num_rows > 0:
        print(f"Processing a sample of {args.num_rows} rows...")
        df = df.head(args.num_rows)
    else:
        print("Processing all rows...")

    # Scrape the text for each URL and add it to a new column
    tqdm.pandas(desc="Scraping Articles")
    df['scraped_text'] = df['Url'].progress_apply(scrape_text_v1)
    
    # Determine S3 object key
    if args.s3_key:
        s3_object_key = args.s3_key
    elif args.num_rows is not None and args.num_rows > 0:
        s3_object_key = f'processed_data/{BASE_OUTPUT_FILENAME}_sample_{args.num_rows}.csv'
    else:
        s3_object_key = f'processed_data/{BASE_OUTPUT_FILENAME}_full.csv'
    
    # Upload the DataFrame directly to S3
    upload_df_to_s3(df, S3_BUCKET_NAME, s3_object_key, AWS_REGION)
    
    print("--- First 5 rows of the result ---")
    print(df.head())


if __name__ == '__main__':
    main()
