
#!/usr/bin/env python

"""
Script split the large consumer complaint records into chunks
to demonstrate the demo model pipeline.
Each data chunck will contain all records between the start date
of the records (2011-12-01) and the end date
"""

import re
import string
import pandas as pd

# TODO: slugify all headers
# TODO: provide name for id columns and columns without titles


def _update_header(x):
    x = x.replace(" ", "_")
    x = x.replace("-", "_")
    x = x.replace("?", "")
    # x = re.sub(r'\W_+', '', x)
    return x


def update_headers(df):
    df.columns = [_update_header(i) for i in df.columns]


def read_data(path='../data/Consumer_complaints.csv'):
    print("Reading the data ...")
    df = pd.read_csv(path)
    df = df.replace(r'\s', ' ', regex=True)
    df["Issue"] = df["Issue"].replace(r',', ' ', regex=True)
    df['Date received'] = pd.to_datetime(df['Date received'])
    # df = df.sort_values('Date received')
    update_headers(df)
    return df


def split_by_date(df, end_date='2011-12-07'):
    # start date 2011-12-01
    # end date 2017-11-15
    return df.loc[df['Date_received'] <= end_date]


def save_data(df, path):
    df.to_csv(path, index_label="id")
    print(f"Saved {df.count()['Date_received']} records to {path}")


if __name__ == "__main__":

    end_date = "2011-12-31"
    df = read_data()
    sub_df = split_by_date(df, end_date)
    save_data(sub_df, f"../airflow/data/pipeline_data.csv")