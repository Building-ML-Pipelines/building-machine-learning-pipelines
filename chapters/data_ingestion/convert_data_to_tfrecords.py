""" Example module to convert csv data to TFRecords
"""

import csv

import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode()])
    )


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def clean_rows(row):
    if not row["zip_code"]:
        row["zip_code"] = "99999"
    return row

def convert_zipcode_to_int(zipcode):
    if isinstance(zipcode, str) and "XX" in zipcode:
        zipcode = zipcode.replace("XX", "00")
    int_zipcode = int(zipcode)
    return int_zipcode
        

original_data_file = "../../data/consumer_complaints_with_narrative.csv"
tfrecords_filename = "consumer-complaints.tfrecords"
tf_record_writer = tf.io.TFRecordWriter(tfrecords_filename)

with open(original_data_file) as csv_file:
    reader = csv.DictReader(csv_file, delimiter=",", quotechar='"')
    for row in tqdm(reader):
        row = clean_rows(row)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "product": _bytes_feature(row["product"]),
                    "sub_product": _bytes_feature(row["sub_product"]),
                    "issue": _bytes_feature(row["issue"]),
                    "sub_issue": _bytes_feature(row["sub_issue"]),
                    "state": _bytes_feature(row["state"]),
                    "zip_code": _int64_feature(convert_zipcode_to_int(row["zip_code"])),
                    "company": _bytes_feature(row["company"]),
                    "company_response": _bytes_feature(row["company_response"]),
                    "timely_response": _bytes_feature(row["timely_response"]),
                    "consumer_disputed": _bytes_feature(
                        row["consumer_disputed"]
                    ),
                }
            )
        )
        tf_record_writer.write(example.SerializeToString())
    tf_record_writer.close()
