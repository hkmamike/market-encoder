# Install Python3 - instructions TODO

# Setup virtual env

```
python3 -m venv venv-cs230
source venv-cs230/bin/activate
pip install -r requirements.txt
```

Note: if you installed new packages, run `pip freeze > requirements.txt` to update requirements.txt so others don't need to manually repeat it

## Environment Verification - run the example model file to make sure it works

```
python3 fullRun.py     # don't check in the resulting files into github
```

# Configure AWS Profile

```
$ aws configure --profile team-s3-uploader
```

It will prompt you for four things. Use the keys you just saved: 
AWS Access Key ID,
AWS Secret Access Key, etc.

For our team, this is in CS230 Tracker -> S3 bucket access tab

# Configure AWS Profile Credentials Debugging

```
cat ~/.aws/credentials
cat ~/.aws/config
```

# Files Overview

* `fullRun.py`: Unrelated to project, given by Section 3 TA, used to make sure the environment works as expected
* `huggingface_insight.py`: Modify/run this script to check what is available on hugging face dataset
* `scrape_news.py`: Downloads the hugging face dataset with specified split (X%, 0:100, etc.) and also produces only first Y rows of article text as specified
* `financial_news_with_text_sample`: Produced by scrape_news.py
* `upload_s3.py`: Once we have satisfactory csv, we can upload to s3

# Market encoder

```
python scrape_news.py --num-rows 10
```                                         

This code will download the data from huggingface (~30GB) and will start scraping. It uses a local cache so it won't need to download the data again on script rerun. `--num-rows` controls how many rows of articles to scrape. If absent it scrapes everything output is csv.

