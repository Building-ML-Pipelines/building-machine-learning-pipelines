
#!/usr/bin/env python3

"""
Downloads the csv data
"""

import csv
import os
import urllib3
import shutil

import logging


# Initial dataset source
DATASET_URL = "http://bit.ly/building-ml-pipelines-dataset"

# Initial local dataset location
LOCAL_FILE_NAME = "data/tmp-consumer-complaints.csv"

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

def _update_col_names(x, i):
    """Internal helper function to convert the names of the initial dataset headers

    Keyword Arguments:
        x {string} -- name of the column (can be None)
        i {integer} -- integer representing the number of the column
    Returns:
        string - returns simplified string version of the column.
                 If the column didn't have a name, it return "col_{number of the column}"
    """
    if x != "":
        x = x.replace(" ", "_")
        x = x.replace("-", "_")
        x = x.replace("?", "")
    else:
        x = f"col_{i}"
    return x.lower()


def clean_zip_code(x):
    if x:
        return x.replace("XX", "00")
    else:
        return '99999'


def update_headers():
    """update_headers updates the header row of the csv file and write the entire file to a new
    file with the file name appendix "_modified.csv"

    Keyword Arguments:
        None
    Returns:
        None
    """

    modified_file_name = os.path.splitext(LOCAL_FILE_NAME)[0].replace("tmp-", "") + ".csv"

    with open(LOCAL_FILE_NAME, 'r', newline='', encoding='utf8') as input_file, open(modified_file_name, 'w', newline='', encoding='ascii', errors='ignore') as output_file:
        r = csv.reader(input_file)
        w = csv.writer(output_file)

        header = next(r, "")  # update the header row
        new_header = [_update_col_names(col_name, i) for i, col_name in enumerate(header)]
        w.writerow(new_header)

        # copy the rest
        zipcode_idx = new_header.index('zip_code')
        for row in r:
            row[zipcode_idx] = clean_zip_code(row[zipcode_idx])
            w.writerow(row)
        logging.info(f"CSV header updated and rewriten to {modified_file_name}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info('Started')

    if check_execution_path():
        create_folder()
        download_dataset()
        update_headers()
        os.remove(LOCAL_FILE_NAME)

    logging.info('Finished')