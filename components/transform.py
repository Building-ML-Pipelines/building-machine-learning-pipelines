import tensorflow as tf 
import tensorflow_transform as tft


_ONE_HOT_FEATURE_KEYS = [
    "product", "sub_product", "company_response", "state", "issue"
]

_ONE_HOT_FEATURE_DIMS = [11, 45, 5, 60, 90]

# buckets for zip_code
_FEATURE_BUCKET_COUNT = 10

_TEXT_FEATURE_KEYS = ["consumer_complaint_narrative"]

_LABEL_KEY = "consumer_disputed"


def _transformed_name(key):
    return key + '_xf'

def _fill_in_missing(x, to_string=False, force_zero=False):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string or to_string else 0
    
    if force_zero:
        default_value = '0'

    if type(x) == tf.SparseTensor:
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]), default_value)
    return tf.squeeze(x, axis=1)


def preprocess_text(text):
    """
    docs go here
    """
    # let's lower all input strings and remove unnecessary characters
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r" '| '|^'|'$", " ")
    text = tf.strings.regex_replace(text, "[^a-z' ]", " ")
    return text


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
    docs go here
    """
    zipcode = tf.strings.regex_replace(zipcode, r'X|\[|\*|\+|\-|`|\.|\ |\$|\/|!|\(', "0")
    zipcode = tf.strings.to_number(zipcode, out_type=tf.dtypes.float32)
    
    return zipcode


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for i, key in enumerate(_ONE_HOT_FEATURE_KEYS):
        int_value = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key], to_string=True),
            top_k=_ONE_HOT_FEATURE_DIMS[i] + 1)
        outputs[_transformed_name(key)] = convert_num_to_one_hot(
            int_value,
            num_labels=_ONE_HOT_FEATURE_DIMS[i] + 1
        )

    # specific to this column:
    temp_zipcode = tft.bucketize(
            convert_zip_code(_fill_in_missing(inputs["zip_code"], force_zero=True)),
            _FEATURE_BUCKET_COUNT,
            always_return_num_quantiles=False)
    outputs[_transformed_name("zip_code")] = convert_num_to_one_hot(
            temp_zipcode,
            num_labels=_FEATURE_BUCKET_COUNT + 1)
        
    for key in _TEXT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = preprocess_text(
            _fill_in_missing(inputs[key], to_string=True)
        )

    outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]
        
    return outputs

