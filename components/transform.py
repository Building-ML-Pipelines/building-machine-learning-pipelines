import tensorflow as tf 
import tensorflow_transform as tft


ONE_HOT_FEATURES = (
    # feature name, feature dimensionality
    ("product", 11),
    ("sub_product", 45),
    ("company_response", 5), 
    ("state", 60),
    ("issue", 90)
)

ONE_HOT_FEATURE_KEYS = [x[0] for x in ONE_HOT_FEATURES]
ONE_HOT_FEATURE_DIMS = [x[1] for x in ONE_HOT_FEATURES]

# buckets for zip_code
FEATURE_BUCKET_COUNT = 10

TEXT_FEATURE_KEYS = ["consumer_complaint_narrative"]

LABEL_KEY = "consumer_disputed"


def transformed_name(key):
    return key + '_xf'

def fill_in_missing(x, to_string=False):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string or to_string else 0

    if type(x) == tf.SparseTensor:
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]), default_value)
    return tf.squeeze(x, axis=1)


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
        int: num_labels (default is 2) 
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

    for i, key in enumerate(ONE_HOT_FEATURE_KEYS):
        int_value = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key], to_string=True),
            top_k=ONE_HOT_FEATURE_DIMS[i] + 1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value,
            num_labels=ONE_HOT_FEATURE_DIMS[i] + 1
        )

    # specific to this column:
    temp_zipcode = tft.bucketize(
            convert_zip_code(fill_in_missing(inputs["zip_code"])),
            FEATURE_BUCKET_COUNT,
            always_return_num_quantiles=False)
    outputs[transformed_name("zip_code")] = convert_num_to_one_hot(
            temp_zipcode,
            num_labels=FEATURE_BUCKET_COUNT + 1)
        
    for key in TEXT_FEATURE_KEYS:
        outputs[transformed_name(key)] = \
            fill_in_missing(inputs[key], to_string=True)

    outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]
        
    return outputs

