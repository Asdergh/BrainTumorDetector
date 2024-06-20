import numpy as np
import tensorflow as tf



class ParallelModel:

    def __init__(self, filters_size, learning_rate, pooling_size, 
                 image_size=(224, 224)) -> None:

        self.model_type = "PARALLEL"
        self.filters_size = filters_size
        self.learning_rate = learning_rate
        self.pooling_size = pooling_size
        self.image_size = image_size

        self.first_model_input_tensor = tf.keras.Input(shape=(self.image_size[0], self.image_size[1], 3))
        self.second_model_input_tensor = tf.keras.Input(shape=(self.image_size[0], self.image_size[1], 3))


        self.first_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_input_tensor)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.first_model_layer)


        self.first_model_layer = tf.keras.layers.Flatten()(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Dense(100, activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Dense(64, activation="relu")(self.first_model_layer)
        self.first_model_layer = tf.keras.layers.Dense(32, activation="relu")(self.first_model_layer)
        self.first_model_last_layer = tf.keras.layers.Dense(1, activation="sigmoid")(self.first_model_layer)


        self.second_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_input_tensor)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Conv2D(32, (self.filters_size[0], self.filters_size[1]), activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.MaxPooling2D((self.pooling_size[0], self.pooling_size[1]))(self.second_model_layer)


        self.second_model_layer = tf.keras.layers.Flatten()(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Dense(100, activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Dense(64, activation="relu")(self.second_model_layer)
        self.second_model_layer = tf.keras.layers.Dense(32, activation="relu")(self.second_model_layer)
        self.second_model_last_layer = tf.keras.layers.Dense(1, activation="sigmoid")(self.second_model_layer)


        self.first_model = tf.keras.Model(self.first_model_input_tensor, self.first_model_last_layer)
        self.second_model = tf.keras.Model(self.second_model_input_tensor, self.second_model_last_layer)

        self.first_model.compile(
             optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate),
             loss=tf.keras.losses.BinaryCrossentropy(),
             metrics=["mae"]
        )

        self.second_model.compile(
             optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
             loss=tf.keras.losses.BinaryCrossentropy(),
             metrics=["accuracy"]
        )

        self.models = [self.first_model, self.second_model]
    
    def set_weights(self, weights_collection):

        for model in self.models:

            for (layers_number, layer) in weights_collection.keys():

                layer_weigths = weights_collection[layer]["weigths"]
                layer_biases = weights_collection[layer]["biases"]
                model.layers[layers_number].set_weights([layer_weigths, layer_biases])
    
    def get_curent_weights(self):

        weigths_collection = {}
        for (model_number, model) in enumerate(self.models):
            
            for layer in model.layers:
                
                curent_weights = layer.get_weigths()
                weigths_collection[f"model_{model_number}"]["weights"] = curent_weights[0]
                weigths_collection[f"model_{model_number}"]["biases"] = curent_weights[1]
        
        return weigths_collection

            



