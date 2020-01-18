
#!/usr/bin/env python3

"""
Downloads the csv data 
"""

import os
import urllib3
import shutil

import logging


DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/26k-consumer-complaints.csv"

def download_dataset(url=DATASET_URL):
    c = urllib3.PoolManager()
    filename = "data/26k-consumer-complaints.csv"

    with c.request('GET', url, preload_content=False) as res, open(filename, 'wb') as out_file:
        shutil.copyfileobj(res, out_file)
    logging.info("Download completed.")


def create_folder():
    directory = "data/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Data folder created.")


def check_execution_path():
    file_name = "LICENSE"
    if not os.path.exists(file_name):
        logging.error(
            "Don't execute the script from a sub-directory. "
            "Switch to the root of the project folder")
        return False
    return True

if __name__ == "__main__":

    if check_execution_path():
        create_folder()
        download_dataset()