#!/usr/bin/env python3

"""
Downloads the csv data
"""

import logging
import os
import shutil

import pandas as pd
import urllib3

# Initial dataset source
DATASET_URL = "http://bit.ly/building-ml-pipelines-dataset"

# Initial local dataset location
LOCAL_FILE_NAME = "data/tmp_consumer_complaints_with_narrative.csv"


def download_dataset(url=DATASET_URL):
    """download_dataset downloads the remote dataset to a local path

    Keyword Arguments:
        url {string} --
            complete url path to the csv data source (default: {DATASET_URL})
        local_path {string} --
            initial local file location (default: {LOCAL_FILE_NAME})
    Returns:
        None
    """
    # disable insecure https warning
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    c = urllib3.PoolManager()
    with c.request("GET", url, preload_content=False) as res, open(
        LOCAL_FILE_NAME, "wb"
    ) as out_file:
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
    """Check if the function and therefore all subsequent functions
        are executed from the root of the project

    Returns:
        boolean -- returns False if execution path isn't the root,
            otherwise True
    """
    file_name = "LICENSE"
    if not os.path.exists(file_name):
        logging.error(
            "Don't execute the script from a sub-directory. "
            "Switch to the root of the project folder"
        )
        return False
    return True


def update_csv():
    """update_csv updates the header row of the csv file, preprocesses
        the data and writes the entire file to a new file with the file name
        appendix "with_narrative.csv"

    Keyword Arguments:
        None
    Returns:
        None
    """

    file_name_part = os.path.splitext(LOCAL_FILE_NAME)[0]
    modified_file_name = file_name_part.replace("tmp_", "") + ".csv"

    feature_cols = [
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "consumer_complaint_narrative",
        "company",
        "state",
        "zip_code",
        "company_response",
        "timely_response",
        "consumer_disputed",
    ]
    df = pd.read_csv(LOCAL_FILE_NAME, usecols=feature_cols)

    df = df[df["consumer_complaint_narrative"].notnull()]
    df = df.sample(frac=1, replace=False).reset_index(drop=True)
    df["zip_code"] = df["zip_code"].str.replace("XX", "00")

    df.to_csv(modified_file_name, index=False)
    logging.info(f"CSV header updated and rewritten to {modified_file_name}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Started")

    if check_execution_path():
        create_folder()
        download_dataset()
        update_csv()
        os.remove(LOCAL_FILE_NAME)

    logging.info("Finished")
