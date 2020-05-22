import tensorflow as tf 
import tensorflow_transform as tft


ONE_HOT_FEATURES = {
    "product": 11,
    "sub_product": 45,
    "company_response": 5, 
    "state": 60,
    "issue": 90
}

# feature name, bucket count
BUCKET_FEATURES = {
    "zip_code": 10
}

# feature name, value is unused
TEXT_FEATURES = {
    "consumer_complaint_narrative": None
}

LABEL_KEY = "consumer_disputed"


def transformed_name(key):
    return key + '_xf'

def fill_in_missing(x):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    if isinstance(x, tf.SparseTensor):
        x = tf.sparse.to_dense(x, default_value)
    return tf.reshape(x, shape=[-1])


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def convert_zip_code(zipcode):
    """
    Convert a zipcode string to int64 representation. In the dataset the
    zipcodes are anonymized by repacing the last 3 digits to XXX. We are 
    replacing those characters to 000 to simplify the bucketing later on.

    Args:
        str: zipcode
    Returns:
        zipcode: int64
    """
    if zipcode == '':
        zipcode = "00000"
    zipcode = tf.strings.regex_replace(zipcode, r'X{0,5}', "0")
    zipcode = tf.strings.to_number(zipcode, out_type=tf.float32)
    return zipcode



def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in ONE_HOT_FEATURES.keys():
        dim = ONE_HOT_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k=dim + 1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1)

    for key, bucket_count in BUCKET_FEATURES.items():
        temp_feature = tft.bucketize(
                convert_zip_code(fill_in_missing(inputs[key])),
                bucket_count,
                always_return_num_quantiles=False)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
                temp_feature,
                num_labels=bucket_count + 1)
        
    for key in TEXT_FEATURES.keys():
        outputs[transformed_name(key)] = \
            fill_in_missing(inputs[key])

    outputs[transformed_name(LABEL_KEY)] = fill_in_missing(inputs[LABEL_KEY])
        
    return outputs
