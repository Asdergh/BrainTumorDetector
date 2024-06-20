import numpy as np
import tensorflow as tf


class ConcatenativeModel:

    def __init__(self, learning_rate, filters_size, 
                 pooling_size=None, pooling_type=None, 
                 image_size=(224, 224)) -> None:
        
        self.learning_rate = learning_rate
        self.filters_size = filters_size
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type
        self.image_size = image_size

        self.model_type = "CONCATINATIVE"
        if self.pooling_size is None:
            self.pooling_size = (self.filters_size - 1, self.filters_size - 1)
        
        self.input_tensor_1 = tf.keras.Input(shape=(self.image_size[1], self.image_size[0], 3))
        self.input_tensor_2 = tf.keras.Input(shape=(self.image_size[1], self.image_size[0], 3))

        self.first_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.input_tensor_1)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(64, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(64, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(126, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(126, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)

        self.second_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.input_tensor_2)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(64, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(64, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(126, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(126, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)

        self.concatenated_layer = tf.keras.layers.concatenate([self.first_model_layer, self.second_model_layer], axis=-1)
        self.model_classification_layer = tf.keras.layers.Flatten()(self.concatenated_layer)
        self.model_classification_layer = tf.keras.layers.Dense(100, activation="relu")(self.model_classification_layer)
        self.model_classification_layer = tf.keras.layers.Dense(32, activation="relu")(self.model_classification_layer)
        self.model_classification_layer = tf.keras.layers.Dense(1, activation="relu")(self.model_classification_layer)

        self.model = tf.keras.Model([self.input_tensor_1, self.input_tensor_2], self.model_classification_layer)
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
    
    def set_weigths(self, weights_collection):

        for (layer_number, layer) in enumerate(weights_collection.keys()):

            curent_weigths = weights_collection[layer]["weights"]
            curent_biases = weights_collection[layer]["biases"]
            self.model.layers[layer_number].set_weigths([curent_weigths, curent_biases])
        
    def get_weigths(self):
        
        weights_collection = {}
        for (layer_number, layer) in self.models.layers:
            
            all_weights = layer.get_weigths()
            weights_collection[f"layer_{layer_number}"]["weigths"] = all_weights[0]
            weights_collection[f"layer_{layer_number}"]["biases"] = all_weights[1]
        
        return weights_collection