"""
This module implements a standard feed forward model with a variable number of
parameters and optimized with Adam - the most popular variant of Stochastic
Gradient Decent.

"""
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, concatenate
from nn_models.base_nn_model import BaseNNModel
import numpy as np

class Model01(BaseNNModel):
    """
    This class implements the Model described in the module description.
    """

    def __init__(self, input_count, hidden_layer_sizes=[10, 20, 10]):
        super().__init__()

        x_input = Input(shape=(input_count,), name="input_layer")

        x = x_input
        for hidden_layer_size in hidden_layer_sizes:
            x = Dense(hidden_layer_size, activation='relu')(x)
            x = Dropout(0.5)(x)

        x_output = Dense(1, activation='sigmoid', name="output")(x)

        self.internal_model = Model(inputs=[x_input], outputs=[x_output])
        self.internal_model.compile(optimizer='nadam', loss='mse')

    @property
    def name(self):
        return "Model01"

    def fit(self, inputs, outputs, epochs=1, verbose=1):
        return self._internal_keras_model.fit(inputs,
                                              outputs,
                                              epochs=epochs,
                                              verbose=verbose)

    def predict(self, inputs):
        return np.round(self._internal_keras_model.predict(inputs))

    @property
    def metric_names(self):
        return self._internal_keras_model.metrics_names

    def evaluate(self, inputs, true_outputs):
        pred_outputs = self.predict(inputs)
        return np.mean(pred_outputs == true_outputs)


