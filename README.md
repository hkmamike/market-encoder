# Install Python3 - instructions TODO

# Setup virtual env
python3 -m venv venv-cs230
source venv-cs230/bin/activate
pip install -r requirements.txt

Note: if you installed new packages, run "pip freeze > requirements.txt" to update requirements.txt so others don't need to manually repeat it

## Environment Verification - run the example model file to make sure it works
python3 fullRun.py     # don't check in the resulting files into github

# Configure AWS Profile
aws configure --profile team-s3-uploader

It will prompt you for four things. Use the keys you just saved: AWS Access Key ID, AWS Secret Access Key, etc.

# Files Overview
fullRun.py                          <- unrelated to project, given by Section 3 TA, used to make sure the environment works as expected

huggingface_insight.py              <- modifying/run this script to check what is available on hugging face dataset
scrape_news.py                      <- downloads the hugging face dataset with specified split (X%, 0:100, etc.) and also produce only first Y rows of article text as specified
financial_news_with_text_sample     <- produced by scrape_news.py

upload_s3.py                        <- once we have satisfactory csv, we can upload to s3

# market encoder
python scrape_news.py --num-rows 10      # this code will download the data from huggingface (~30GB) and would start scraping
                                         # it uses a locally cache so it wouldn't need to download the data again on script rerun
                                         # num-rows controls how many rows of articles to scrape, if absent it scrapes everything
                                         # output is csv

# Setup - Gemini Cli (MacOS) - not working yet

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Follow terminal instructions after installing to add brew path

python3 -m pip install virtualenv

brew install google-cloud-sdk

gcloud init
