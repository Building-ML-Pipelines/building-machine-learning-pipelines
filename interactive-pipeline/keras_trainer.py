import tensorflow as tf 
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

#from .model import convert_model_to_estimator, get_model
#from .transform import _transformed_name, _LABEL_KEY

import tensorflow_hub as hub

def _transformed_name(key):
    return key + '_xf'

_LABEL_KEY = 'consumer_disputed'

def get_model(show_summary=True):
    """
    Function defines a Keras model and returns the model as Keras object
    """
    # how do you get the size of the embedding?? Why use an embedding when there are only 2 dimensions to start with?
    # in fact, why 2 dimensions for 0/1?

    # "product", "sub_product", "issue", "sub_issue", "state", "zip_code", "company", "company_response", "timely_response"
    #    x0            x1           x2        x3         x4        x5          x6             x7                 x8

    # Cat: "product", "sub_product", "state", "zip_code", "company_response", "timely_response", "consumer_disputed"

    # one-hot categorical features
    num_products = 11
    num_company_responses = 6
    num_timely_responses = 2

    input_product = tf.keras.Input(shape=(num_products + 1,), name="product_xf")
    input_company_response = tf.keras.Input(shape=(num_company_responses + 1,), name="company_response_xf")
    input_timely_response = tf.keras.Input(shape=(num_timely_responses + 1,), name="timely_response_xf")

    # categorical features
    input_sub_product = tf.keras.Input(shape=(1,), name="sub_product_xf")
    input_state = tf.keras.Input(shape=(1,), name="state_xf")
    input_zip_code = tf.keras.Input(shape=(1,), name="zip_code_xf")

    # text features
    module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
    embed = hub.KerasLayer(module_url)
    input_issue = tf.keras.Input(shape=(1,), name="issue_xf", dtype=tf.string)
    input_sub_issue = tf.keras.Input(shape=(1,), name="sub_issue_xf", dtype=tf.string)
    input_company_name = tf.keras.Input(shape=(1,), name="company_xf", dtype=tf.string)

    def cnn_layers(x):
        reshaped_x = tf.reshape(x, [-1])
        embedded_x = embed(reshaped_x) 
        conv_x = tf.keras.layers.Reshape((1, 128), input_shape=(128,))(embedded_x)
        conv_x = tf.keras.layers.Conv1D(32, 1, activation='relu')(conv_x)
        conv_x = tf.keras.layers.GlobalMaxPooling1D()(conv_x)
        conv_x = tf.keras.layers.Dense(10, activation='relu')(conv_x)
        return conv_x

    x0 = input_product
    x7 = input_company_response
    x8 = input_timely_response

    # convert to embeddings
    x1 = tf.keras.layers.Embedding(70, 5)(input_sub_product)
    x1 = tf.keras.layers.Reshape((5, ), input_shape=(1, 5))(x1)

    x4 = tf.keras.layers.Embedding(70, 5)(input_state)
    x4 = tf.keras.layers.Reshape((5, ), input_shape=(1, 5))(x4)

    x5 = tf.keras.layers.Embedding(10000, 5)(input_zip_code)
    x5 = tf.keras.layers.Reshape((5, ), input_shape=(1, 5))(x5)

    x_feed_forward = tf.keras.layers.concatenate(
        [x0, x1, x4, x5, x7, x8])

    x = tf.keras.layers.Dense(100, activation='relu')(x_feed_forward)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)

    conv_issue = cnn_layers(input_issue)
    conv_sub_issue = cnn_layers(input_sub_issue)
    conv_company = cnn_layers(input_company_name)

    x = tf.keras.layers.concatenate([x, conv_company, conv_issue, conv_sub_issue])
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x) 

    _inputs = [input_product, input_company_response, input_timely_response, 
              input_sub_product, input_state, input_zip_code, 
              input_issue, input_sub_issue, input_company_name] 

    keras_model = tf.keras.models.Model(_inputs, output)
    keras_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',  # categorical_crossentropy
                     metrics=[
                         'accuracy', 
                         tf.keras.metrics.TruePositives(), 
                         tf.keras.metrics.TrueNegatives(), 
                         tf.keras.metrics.FalsePositives(), 
                         tf.keras.metrics.FalseNegatives()
                         ])
    if show_summary:
        keras_model.summary()

    return keras_model

def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = tf_transform_output.transform_raw_features(
            parsed_features)
        transformed_features.pop(_transformed_name(_LABEL_KEY))

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _input_fn(file_pattern,#: Text,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
    file_pattern: input tfrecord file pattern.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

      Returns:
        A dataset that contains (features, indices) tuple where features is a
          dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=_transformed_name(_LABEL_KEY))

    return dataset


# TFX Trainer will call this function.
def run_fn(fn_args):#: TrainerFnArgs):
    """Train the model based on given args.

    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

    model = get_model()

    model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

    signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)