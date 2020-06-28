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
"""TFX complaint model model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List

# At least one feature is needed.

# feature name, feature dimensionality
ONE_HOT_FEATURES = {
    "product": 11,
    "sub_product": 45,
    "company_response": 5,
    "state": 60,
    "issue": 90,
}

# feature name, bucket count
BUCKET_FEATURES = {"zip_code": 10}

# feature name, value is unused
TEXT_FEATURES = {"consumer_complaint_narrative": None}

# Keys
LABEL_KEY = "consumer_disputed"


def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + "_xf"


def vocabulary_name(key: Text) -> Text:
    """Generate the name of the vocabulary feature from original name."""
    return key + "_vocab"


def transformed_names(keys: List[Text]) -> List[Text]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]
