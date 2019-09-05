"""
Module for Experiment 01
"""
from experiments.base_experiment import Experiment

class Experiment01(Experiment):
    """
    Main Experiment
    """

    def _experiment(self):
        # Including imports to get accurate time of library inclusion when running experiment.
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import scale
        from sklearn.metrics import classification_report
        import settings
        import random
        import numpy as np
        from data_handlers.data_extractor import DataExtractor
        from data_handlers.data_articulation_loaders import DataArticulationLoaders
        from nn_models.fully_connected.model01 import Model01


        self.log("Training Data Loading Started via DataLoader")
        data_models = DataExtractor.extract_data()
        self.log("Training Data Loaded via DataLoader")

        inputs = DataArticulationLoaders.get_standard_predictors(data_models)

        # Scale the inputs so they have a mean of 0 and variance of 1
        inputs = scale(inputs)
        outputs = DataArticulationLoaders.get_is_baetis_present(data_models)

        number_of_predictors = inputs.shape[1]

        train_percentage_of_sample = 0.8

        x_train, x_test, y_train, y_test = train_test_split(inputs,
                                                            outputs,
                                                            train_size=train_percentage_of_sample,
                                                            random_state=settings.SEED)

        model = Model01(input_count=number_of_predictors, hidden_layer_sizes=[1])
        self.log("Model used in this experiment: " + str(model.name))
        self.log("Saving Neural Network Model Visualization")
        model.save_model_visualization()
        self.log("Model beginning training")

        model.fit(x_train, y_train, epochs=1, verbose=10)

        self.log("Model finished training")

        train_prediction_results = model.predict(x_train)
        train_evalutation_results = model.evaluate(x_train, y_train)
        self.log("Results of Model when inputs included samples seen before (Train % Correct):" + str(train_evalutation_results))
        self.log("Prevalence of Presences: " + str(np.mean(y_train)))
        self.log("Classification Report: \n" + str(classification_report(train_prediction_results, y_train)))

        test_prediction_results = model.predict(x_test)
        test_evalutation_results = model.evaluate(x_test, y_test)
        self.log("Results of Model when inputs included samples seen before (Train% Correct):" + str(test_evalutation_results))
        self.log("Prevalence of Presences: " + str(np.mean(y_test)))
        self.log("Classification Report: \n" + str(classification_report(test_prediction_results, y_test)))

        """
        self.log("Total Number of Samples Loaded: " + str(len(data_models_for_training)))
        self.log("Training Percentage of Samples: " + str(train_percentage_of_sample))

        inputs, outputs = DataArticulationLoaders.schwartz_multi_channel_individually_logged_norm_error_norm_with_augmentation_with_attitude(data_models_for_training)
        self.log("Data articulated for nn model")
        x_train, x_test, y_train, y_test = train_test_split(inputs,
                                                            outputs,
                                                            train_size=train_percentage_of_sample,
                                                            random_state=settings.SEED)
        self.log("x_train length:" + str(len(x_train)))

        model = Model12D()
        self.log("Model used in this experiment: " + str(model.name))
        self.log("Saving Neural Network Model Visualization")
        model.save_model_visualization()
        self.log("Model beginning training")

        model.fit(x_train, y_train, epochs=1, verbose=10)
        self.log("Model finished training")

        self.log("Output Layer Weights: " + str(model.output_layer.get_weights()))

        test_results = model.evaluate(x_test, y_test)
        self.log("Results of Model when inputs included targets seen before (MSE):" + str(test_results))

        self.log("Evaluation Data Loading Started via DataLoader")
        data_models_for_evaluation = DataExtractor.extract_nacho_data(list_of_target_numbers=evaluation_target_numbers)
        self.log("Evaluation Data Loaded via DataLoader")

        eval_inputs, eval_outputs = DataArticulationLoaders.schwartz_multi_channel_individually_logged_norm_error_norm_with_augmentation_with_attitude(data_models_for_evaluation)
        eval_results = model.evaluate(eval_inputs, eval_outputs)
        self.log("Results of Model when inputs did not include targets seen before (MSE):" + str(eval_results))
        """
