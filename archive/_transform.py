import tensorflow as tf 
import tensorflow_transform as tft


_ONE_HOT_FEATURE_KEYS = [
    "product", "company_response", "timely_response"
]

_ONE_HOT_FEATURE_DIMS = [11, 6, 2]

_OOV_SIZE = 10

_CATEGORICAL_FEATURE_KEYS = [
    "sub_product", "state", "zip_code", 
]

_MAX_CATEGORICAL_FEATURE_VALUES = [41, 59, 90000] 

_TEXT_FEATURE_KEYS = [
    "issue", "sub_issue", "company"
]

_LABEL_KEY = 'consumer_disputed'


def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _fill_in_missing(x, to_string=False, unk=""):
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


def preprocessing_text(text):
    # let's lower all input strings
    text = tf.strings.lower(text)
    
    # Before applying the word piece tokenization, let's remove unnecessary 
    # tokens. This regex_replace was suggested by Aurélien Geron in his 
    # TFX workshop at TensorFlow World 2019
    # https://github.com/tensorflow/workshops/blob/master/tfx_labs/Lab_10_Neural_Structured_Learning.ipynb
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

def convert_label(x):
    keys_tensor = tf.constant(['No', '', 'Yes'])
    vals_tensor = tf.constant([0, 0, 1])
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
    converted_label = table.lookup(x)
    converted_label = tf.cast(converted_label, tf.float32)
    converted_label = tf.reshape(converted_label, [-1, 1])
    return converted_label


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for i, key in enumerate(_CATEGORICAL_FEATURE_KEYS):
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key], to_string=True),
            top_k=_MAX_CATEGORICAL_FEATURE_VALUES[i],
            num_oov_buckets=_OOV_SIZE)

    for i, key in enumerate(_ONE_HOT_FEATURE_KEYS):
        int_value = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key], to_string=True),
            top_k=_ONE_HOT_FEATURE_DIMS[i] + 1)
        outputs[_transformed_name(key)] = convert_num_to_one_hot(
            int_value,
            num_labels=_ONE_HOT_FEATURE_DIMS[i] + 1
        )

    for key in _TEXT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = preprocessing_text(
            _fill_in_missing(inputs[key], to_string=True)
        )

    # label conversion
    #outputs[_transformed_name(_LABEL_KEY)] = convert_label(
    #    _fill_in_missing(inputs[_LABEL_KEY], to_string=True))

    outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

    return outputs

