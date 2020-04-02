import tensorflow as tf 
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

import tensorflow as tf
import tensorflow_hub as hub

from typing import List, Text

import os

def get_model(show_summary=True):
    """
    This function defines a Keras model and returns the model as a Keras object.
    """
    
    # one-hot categorical features
    num_products = 12
    num_sub_products = 46
    num_company_responses = 6
    num_states = 61
    num_issues = 91
    num_zip_codes = 11

    input_product = tf.keras.Input(shape=(num_products,), name="product_xf")
    input_sub_product = tf.keras.Input(shape=(num_sub_products,), name="sub_product_xf")
    input_company_response = tf.keras.Input(shape=(num_company_responses,), name="company_response_xf")
    input_state = tf.keras.Input(shape=(num_states,), name="state_xf")
    input_issue = tf.keras.Input(shape=(num_issues,), name="issue_xf")
    input_zip_code = tf.keras.Input(shape=(num_zip_codes,), name="zip_code_xf")

    # text features
    input_narrative = tf.keras.Input(shape=(1,), name="consumer_complaint_narrative_xf", dtype=tf.string)

    # embed text features
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.KerasLayer(module_url)
    reshaped_narrative = tf.reshape(input_narrative, [-1])
    embed_narrative = embed(reshaped_narrative) 
    deep_ff = tf.keras.layers.Reshape((512, ), input_shape=(1, 512))(embed_narrative)
    
    deep = tf.keras.layers.Dense(256, activation='relu')(deep_ff)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(16, activation='relu')(deep)

    wide_ff = tf.keras.layers.concatenate(
        [input_product, input_sub_product, input_company_response, 
         input_state, input_issue, input_zip_code])
    wide = tf.keras.layers.Dense(16, activation='relu')(wide_ff)


    both = tf.keras.layers.concatenate([deep, wide])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(both) 

    _inputs = [input_product, input_sub_product, input_company_response,  
               input_state, input_issue, input_zip_code, 
               input_narrative]

    keras_model = tf.keras.models.Model(_inputs, output)
    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='binary_crossentropy',  
                     metrics=[
                         tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.TruePositives()
                         ])
    if show_summary:
        keras_model.summary()

    return keras_model

_LABEL_KEY = "consumer_disputed"


def _transformed_name(key):
    return key + '_xf'


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


def _input_fn(file_pattern,
              tf_transform_output,
              batch_size= 32):
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
def run_fn(fn_args):
    """Train the model based on given args.

    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')
    
    model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

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