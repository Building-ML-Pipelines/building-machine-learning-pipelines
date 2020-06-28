# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX complaint prediction preprocessing.

This file defines a template for TFX Transform component.
"""

from __future__ import division
from __future__ import print_function

from typing import Union

import tensorflow as tf
import tensorflow_transform as tft

from models import features


def fill_in_missing(x: Union[tf.Tensor, tf.SparseTensor]) -> tf.Tensor:
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a
    dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have
        size at most 1 in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value,
        )
    return tf.squeeze(x, axis=1)


def convert_num_to_one_hot(
    label_tensor: tf.Tensor, num_labels: int = 2
) -> tf.Tensor:
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def convert_zip_code(zipcode: str) -> tf.float32:
    """
    Convert a zipcode string to int64 representation. In the dataset the
    zipcodes are anonymized by repacing the last 3 digits to XXX. We are
    replacing those characters to 000 to simplify the bucketing later on.

    Args:
        str: zipcode
    Returns:
        zipcode: int64
    """
    if zipcode == "":
        zipcode = "00000"
    zipcode = tf.strings.regex_replace(zipcode, r"X{0,5}", "0")
    zipcode = tf.strings.to_number(zipcode, out_type=tf.float32)
    return zipcode


def preprocessing_fn(inputs: tf.Tensor) -> tf.Tensor:
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in features.ONE_HOT_FEATURES.keys():
        dim = features.ONE_HOT_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k=dim + 1
        )
        outputs[features.transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for key, bucket_count in features.BUCKET_FEATURES.items():
        temp_feature = tft.bucketize(
            convert_zip_code(fill_in_missing(inputs[key])),
            bucket_count,
            always_return_num_quantiles=False,
        )
        outputs[features.transformed_name(key)] = convert_num_to_one_hot(
            temp_feature, num_labels=bucket_count + 1
        )

    for key in features.TEXT_FEATURES.keys():
        outputs[features.transformed_name(key)] = fill_in_missing(inputs[key])

    outputs[features.transformed_name(features.LABEL_KEY)] = fill_in_missing(
        inputs[features.LABEL_KEY]
    )

    return outputs
