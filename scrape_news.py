"""
Scrapes financial news articles from URLs provided in a Hugging Face dataset.

This script performs the following steps:
1. Loads the dataset from Hugging Face, relying on the library's built-in cache.
2. Scrapes the text content from the URLs in the dataset.
3. Adds the scraped text as a new column named 'scraped_text'.
4. Saves the processed data to a new CSV file.

Valid Command-Line Arguments:
  --num-rows [INTEGER] : (Optional) The number of rows to process. If not provided,
                           the entire dataset will be processed.

Example Usage:
  # To process a sample of 10 articles for testing
  python scrape_news.py --num-rows 10

  # To process the entire dataset
  python scrape_news.py
"""
import pandas as pd
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse
import os

# --- Configuration ---
DATASET_PATH = "Zihan1004/FNSPID"
BASE_OUTPUT_FILENAME = 'financial_news_with_text'
REQUEST_TIMEOUT = 10
USER_AGENT = 'Mozilla/5.0'

def scrape_text(url):
    """Scrapes the text content from a given URL."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={'User-Agent': USER_AGENT})
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # This is a generic approach; you might need to refine selectors 
        # based on the common structures of the news websites.
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return ""

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
    args = parser.parse_args()

    # --- Load Dataset (Hugging Face handles caching) ---
    print(f"Loading dataset from Hugging Face ({DATASET_PATH})...")
    print("The 'datasets' library will automatically use a local cache if available.")
    dataset = load_dataset(DATASET_PATH, split='train')
    
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    print(f"Loaded {len(df)} articles.")
    
    # --- Limit rows for processing if specified ---
    if args.num_rows is not None and args.num_rows > 0:
        print(f"Processing a sample of {args.num_rows} rows...")
        df = df.head(args.num_rows)
    else:
        print("Processing all rows...")

    # Scrape the text for each URL and add it to a new column
    tqdm.pandas(desc="Scraping Articles")
    df['scraped_text'] = df['url'].progress_apply(scrape_text)
    
    # Determine output filename
    if args.num_rows is not None and args.num_rows > 0:
        output_filename = f'{BASE_OUTPUT_FILENAME}_sample_{args.num_rows}.csv'
    else:
        output_filename = f'{BASE_OUTPUT_FILENAME}.csv'
        
    print(f"\nSaving the updated dataset to {output_filename}...")
    df.to_csv(output_filename, index=False)
    
    print("Scraping and processing complete.")
    print("--- First 5 rows of the result ---")
    print(df.head())


if __name__ == '__main__':
    main()

