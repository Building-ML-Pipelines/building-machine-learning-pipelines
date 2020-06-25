
#!/usr/bin/env python3

"""
Downloads the csv data
"""

import csv
import os
import urllib3
import shutil
import pandas as pd

import logging


# Initial dataset source
DATASET_URL = "http://bit.ly/building-ml-pipelines-dataset"

# Initial local dataset location
LOCAL_FILE_NAME = "data/tmp_consumer_complaints.csv"

def download_dataset(url=DATASET_URL):
    """download_dataset downloads the remote dataset to a local path

    Keyword Arguments:
        url {string} -- complete url path to the csv data source (default: {DATASET_URL})
        local_path {string} -- initial local file location (default: {LOCAL_FILE_NAME})
    Returns:
        None
    """
    c = urllib3.PoolManager()
    with c.request('GET', url, preload_content=False) as res, open(LOCAL_FILE_NAME, 'wb') as out_file:
        shutil.copyfileobj(res, out_file)
    logging.info("Download completed.")


def create_folder():
    """Creates a data folder if it doesn't exist.

    Returns:
        None
    """
    directory = "data/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Data folder created.")
    else:
        logging.info("Data folder already existed.")


def check_execution_path():
    """Check if the function and therefore all subsequent functions are executed
    from the root of the project

    Returns:
        boolean -- returns False if execution path isn't the root, otherwise True
    """
    file_name = "LICENSE"
    if not os.path.exists(file_name):
        logging.error(
            "Don't execute the script from a sub-directory. "
            "Switch to the root of the project folder")
        return False
    return True


def update_csv():
    """update_csv updates the header row of the csv file, preprocesses the data and writes the entire file to a new
    file with the file name appendix "with_narrative.csv"

    Keyword Arguments:
        None
    Returns:
        None
    """

    modified_file_name = os.path.splitext(LOCAL_FILE_NAME)[0].replace("tmp-", "") + "_with_narrative.csv"

    feature_cols=["product", "sub_product", "issue", "sub_issue", "state", "zipcode", "company", "company_response_to_consumer", "timely_response", "consumer_disputed?", "consumer_complaint_narrative"]
    df = pd.read_csv(LOCAL_FILE_NAME, usecols=feature_cols)

    df.columns = df.columns.str.replace(' ','_').str.replace('?', '')
    df = df.rename({'zipcode': 'zip_code', 'company_response_to_consumer': 'company_response'}, axis=1)
    df = df[df['consumer_complaint_narrative'].notnull()]
    df['c'] = df['consumer_disputed'].map({'Yes': 1, 'No': 0})
    df = df.drop('consumer_disputed', axis=1)
    df = df.rename(columns={"c": "consumer_disputed"})
    df = df.sample(frac=1, replace=False).reset_index(drop=True)
    df['zip_code'] = df['zip_code'].str.replace('XX', '00')

    df.to_csv(modified_file_name, index=False)
    logging.info(f"CSV header updated and rewriten to {modified_file_name}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info('Started')

    if check_execution_path():
        create_folder()
        download_dataset()
        update_csv()
        os.remove(LOCAL_FILE_NAME)

    logging.info('Finished')