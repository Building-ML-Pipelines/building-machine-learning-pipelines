"""
This module defines the Keras model
"""

import tensorflow as tf
import tensorflow_hub as hub

def get_model(show_summary=True):
    """
    Function defines a Keras model and returns the model as Keras object
    """
    # how do you get the size of the embedding?? Why use an embedding when there are only 2 dimensions to start with?
    # in fact, why 2 dimensions for 0/1?

    # "product", "sub_product", "issue", "sub_issue", "state", "zip_code", "company", "company_response", "timely_response"
    #    x0            x1           x2        x3         x4        x5          x6             x7                 x8

    # Cat: "product", "sub_product", "state", "zip_code", "company_response", "timely_response", "consumer_disputed"

    # categorical inputs
    input_product = tf.keras.Input(shape=(1,), name="product_xf")
    input_sub_product = tf.keras.Input(shape=(1,), name="sub_product_xf")
    input_state = tf.keras.Input(shape=(1,), name="state_xf")
    input_zip_code = tf.keras.Input(shape=(1,), name="zip_code_xf")
    input_company_response = tf.keras.Input(shape=(1,), name="company_response_xf")
    input_timely_response = tf.keras.Input(shape=(1,), name="timely_response_xf")

    # text inputs
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

    # convert to embeddings
    x0 = tf.keras.layers.Embedding(12, 3)(input_product)
    x0 = tf.keras.layers.Reshape((3, ), input_shape=(1, 3))(x0)

    x1 = tf.keras.layers.Embedding(60, 5)(input_sub_product)
    x1 = tf.keras.layers.Reshape((5, ), input_shape=(1, 5))(x1)

    x4 = tf.keras.layers.Embedding(60, 5)(input_state)
    x4 = tf.keras.layers.Reshape((5, ), input_shape=(1, 5))(x4)

    x5 = tf.keras.layers.Embedding(10000, 5)(input_zip_code)
    x5 = tf.keras.layers.Reshape((5, ), input_shape=(1, 5))(x5)

    x7 = tf.keras.layers.Embedding(10, 3)(input_company_response)
    x7 = tf.keras.layers.Reshape((3, ), input_shape=(1, 3))(x7)

    x8 = tf.keras.layers.Embedding(2, 1)(input_timely_response)
    x8 = tf.keras.layers.Reshape((1, ), input_shape=(1, 2))(x8)

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

    _inputs = [input_product, input_sub_product,
              input_issue, input_sub_issue, 
              input_state, input_zip_code, input_company_name, 
              input_company_response, input_timely_response] 

    keras_model = tf.keras.models.Model(_inputs, output)
    keras_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',  # categorical_crossentropy
                     metrics=[
                         'accuracy', tf.keras.metrics.TruePositives(), 
                         tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), 
                         tf.keras.metrics.FalseNegatives()]
                        )
    if show_summary:
        keras_model.summary()

    return keras_model


def convert_model_to_estimator(keras_model, config):
    """
    function to convert the Keras model to a TensorFlow Estimator
    """
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=keras_model, config=config)
    return estimator