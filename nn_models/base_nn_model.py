"""
Everything to do with how all neural network architectures will work in our
ecosystem.
"""
from abc import ABC, abstractmethod
import keras
import settings
import datetime


class BaseNNModel(ABC):
    """
    Provides an abstract shell which all models must be able to utilize.
    """

    def __init__(self, description=None):
        self._internal_keras_model = None
        self.description = description
        super().__init__()

    @abstractmethod
    def fit(self, inputs, outputs):
        """
        Fits the model using the scikit-learn pattern
        :param inputs: Input training data
        :param outputs: Output training data
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, inputs):
        """
        Gives the output of the trained model given input x.
        :param inputs: Input to predict on.
        :return: The prediction of the model.
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """
        A descriptive, human-readable name of the model.
        :return: The name of the model as a string.
        """
        pass

    @property
    def internal_model(self):
        """
        Allows outward facing access to the internal keras model.
        :return: The internal keras model.
        """
        return self._internal_keras_model

    @internal_model.setter
    def internal_model(self, internal_model):
        """
        Setter for the internal keras model.
        :param internal_model: The internal model to set.
        :return: None
        """
        self._internal_keras_model = internal_model

    def save_model_visualization(self, file_location=None):
        """
        Saves a graphic of the internal keras model to a file.
        :param file_location: The location of the file where the image is to be
        saved.
        :return: None
        """


        import os
        os.environ["PATH"] += os.pathsep + settings.GRAPH_VIZ_LOCATION

        if file_location is None:
            file_location = settings.DEFAULT_MODEL_VISUALIZATION_FOLDER
            file_location += self.name + str(datetime.datetime.now()).replace(":", "-") + ".png"

        keras.utils.plot_model(self.internal_model, to_file=file_location, show_shapes=True)

        """
        # Think about using this code instead.
        import tensorflow as tf
        tf.keras.utils.plot_model(
            self.internal_model,
            to_file=file_location,
            show_shapes=True,
            show_layer_names=True,
            rankdir='LR'
        )
        """