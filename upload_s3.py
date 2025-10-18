"""
Uploads a local file to an Amazon S3 bucket.

This script is designed to be run after `scrape_news.py` has generated a CSV file.

--- Prerequisites ---
1.  An AWS account and an S3 bucket created.
2.  AWS credentials configured locally. The simplest way is to have a file at
    `~/.aws/credentials` (for Linux/macOS) or `%USERPROFILE%\\.aws\\credentials` (for Windows)
    with the following format:

    [default]
    aws_access_key_id = YOUR_ACCESS_KEY
    aws_secret_access_key = YOUR_SECRET_KEY

--- Valid Command-Line Arguments ---
  local_file_path  : (Required) The path to the local file you want to upload.
  s3_object_key    : (Required) The desired name for the file in the S3 bucket (e.g., 'data/financial_news.csv').

--- Example Usage ---
  # To upload a sample file to the S3 bucket
  python upload_to_s3.py financial_news_with_text_sample_10.csv processed_data/financial_news_sample_10.csv

  # To upload the full dataset
  python upload_to_s3.py financial_news_with_text.csv processed_data/financial_news_full.csv
"""
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import argparse
import os

# --- Configuration ---
# TODO: Replace with your specific S3 bucket name and AWS region.
S3_BUCKET_NAME = "your-s3-bucket-name-here"
AWS_REGION = "us-east-1"  # e.g., "us-east-1", "us-west-2"

def upload_to_s3(local_file_path, bucket, s3_object_key):
    """
    Uploads a file to an S3 bucket.

    :param local_file_path: Path to the file to upload.
    :param bucket: The S3 bucket to upload to.
    :param s3_object_key: The desired object key (filename) in S3.
    :return: True if file was uploaded, else False.
    """
    # Check if the local file exists before attempting to upload
    if not os.path.exists(local_file_path):
        print(f"Error: Local file not found at '{local_file_path}'")
        return False

    # Create an S3 client
    # Boto3 will automatically look for credentials in the standard locations.
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    print(f"Uploading '{local_file_path}' to bucket '{bucket}' as '{s3_object_key}'...")
    
    try:
        s3_client.upload_file(local_file_path, bucket, s3_object_key)
        print("Upload Successful!")
        return True
    except FileNotFoundError:
        print(f"Error: The file was not found at {local_file_path}")
        return False
    except (NoCredentialsError, PartialCredentialsError):
        print("Error: AWS credentials not found.")
        print("Please configure your credentials, for example, in ~/.aws/credentials")
        return False
    except ClientError as e:
        # Handle specific boto3 errors, e.g., bucket not found
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"Error: The bucket '{bucket}' does not exist.")
        else:
            print(f"An S3 client error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    """
    Main function to parse arguments and initiate the S3 upload.
    """
    parser = argparse.ArgumentParser(description="Upload a file to Amazon S3.")
    parser.add_argument(
        "local_file_path",
        type=str,
        help="The local path of the file to upload."
    )
    parser.add_argument(
        "s3_object_key",
        type=str,
        help="The desired object key (path/filename) for the file in the S3 bucket."
    )
    args = parser.parse_args()

    # Before running, check if the user has updated the placeholder bucket name
    if S3_BUCKET_NAME == "your-s3-bucket-name-here":
        print("--- CONFIGURATION NEEDED ---")
        print("Please edit the 'upload_to_s3.py' script and replace")
        print("'your-s3-bucket-name-here' with your actual S3 bucket name.")
        return

    upload_to_s3(args.local_file_path, S3_BUCKET_NAME, args.s3_object_key)

if __name__ == '__main__':
    main()

