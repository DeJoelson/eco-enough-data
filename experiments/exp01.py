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
        import numpy as np
        from data_handlers.data_extractor import DataExtractor
        from data_handlers.data_articulation_loaders import DataArticulationLoaders
        from nn_models.fully_connected.model01 import Model01

        """
        Internal Settings
        """

        epochs_to_use = 1
        max_layers_to_have = 5
        max_nodes_in_layer = 100
        min_nodes_in_layer = 2

        self.log("Generating Layers to Evaluate.")
        layer_configurations = self._get_layers_to_evaluate(min_nodes_in_layer, max_nodes_in_layer, max_layers_to_have)
        self.log("Layer Configuration Count: " + str(len(layer_configurations)))


        self.log("Writing Header for the Output CSV File")
        self._write_metric_header_to_csv(max_layers_to_have)

        self.log("Training Data Loading Started via DataLoader")
        data_models = DataExtractor.extract_data()
        self.log("Training Data Loaded via DataLoader")

        inputs = DataArticulationLoaders.get_standard_predictors(data_models)

        # Scale the inputs so they have a mean of 0 and variance of 1
        inputs = scale(inputs)
        data_loader_function_outputs = DataArticulationLoaders.get_is_baetis_present
        outputs = data_loader_function_outputs(data_models)

        number_of_predictors = inputs.shape[1]

        self.log("Number of Predictors: " + str(number_of_predictors))
        self.log("Number of Samples: " + str(len(outputs)))
        self.log("Number of Presences: " + str(sum(outputs)))

        train_percentage_of_sample = 0.8

        x_train, x_test, y_train, y_test = train_test_split(inputs,
                                                            outputs,
                                                            train_size=train_percentage_of_sample,
                                                            random_state=settings.SEED)

        self.log("Number of Samples in Train: " + str(len(y_train)))
        self.log("Number of Presences in Train: " + str(sum(y_train)))

        for index, layer_configuration in enumerate(layer_configurations):
            self.log("Working on number " + str(index) + " of " + str(len(layer_configurations)) + "; " + str(round(((index / len(layer_configurations)) * 100))) + "%")
            model = Model01(input_count=number_of_predictors, hidden_layer_sizes=layer_configuration)
            self.log("Model used in this experiment: " + str(model.name))
            self.log("Saving Neural Network Model Visualization")
            model.save_model_visualization()
            self.log("Model beginning training")

            model.fit(x_train, y_train, epochs=epochs_to_use, verbose=10)

            self.log("Model finished training")

            self.log("Writing Results to CSV")
            self._write_metrics_to_csv(model, epochs_to_use, data_loader_function_outputs, max_layers_to_have, x_train, y_train, x_test, y_test)

    def _get_layers_to_evaluate(self, min_nodes_in_layer, max_nodes_in_layer, max_layers_to_have, max_number_configurations=10000):
        from itertools import permutations
        import numpy as np

        max_values_to_try_per_layer = int(max_number_configurations ** (1/max_layers_to_have))
        acceptable_node_sizes = sorted(set([int(round(value)) for value in np.linspace(min_nodes_in_layer, max_nodes_in_layer, max_values_to_try_per_layer)]))
        layer_configurations = []
        for layer_index in range(max_layers_to_have):
            if len(layer_configurations) > max_number_configurations:
                layer_configurations = layer_configurations[:max_number_configurations]
                break;
            configurations = permutations(acceptable_node_sizes * (layer_index + 1), (layer_index + 1))
            layer_configurations.extend(configurations)

        return [list(_) for _ in layer_configurations]

    def _write_metrics_to_csv(self, model, epochs, response_variable_function, max_hidden_layers, train_x_values,
                              train_y_values, test_x_values, test_y_values):
        number_train = len(train_y_values)
        number_test = len(test_y_values)
        train_test_split = number_train / (number_train + number_test)
        # train_youden, train_precision0, train_roc_auc0, train_recall0, train_f1_score0, train_support0, train_precision1, train_roc_auc1, train_recall1, train_f1_score1, train_support1 =
        # test_youden, test_precision0, test_roc_auc0, test_recall0, test_f1_score0, test_support0, test_precision1, test_roc_auc1, test_recall1, test_f1_score1, test_support1 = self._get_metrics(model, test_x_values, test_y_values)

        hidden_layers = model.hidden_layer_sizes.copy()
        while len(hidden_layers) < max_hidden_layers:
            hidden_layers.append(0)

        row_values = [self.name, response_variable_function, model.name, epochs]
        row_values.extend(hidden_layers)
        row_values.append(train_test_split)
        row_values.extend(self._get_metrics(model, train_x_values, train_y_values))
        row_values.extend(self._get_metrics(model, test_x_values, test_y_values))

        self.add_csv_entry(row_values)

    def _write_metric_header_to_csv(self, max_hidden_layers):
        hidden_layers = ["nodes_in_layer" + str(layer_number) for layer_number in range(max_hidden_layers)]
        while len(hidden_layers) < max_hidden_layers:
            hidden_layers.append("0")

        row_values = ["experiment_name", "response_variable_function", "model_name", "epochs"]
        row_values.extend(hidden_layers)
        row_values.append("train_test_split")
        row_values.extend(
            ["train_youden", "train_roc_auc", "train_precision0", "train_recall0", "train_f1_score0", "train_support0",
             "train_precision1", "train_recall1", "train_f1_score1", "train_support1"])
        row_values.extend(
            ["test_youden", "test_roc_auc" "test_precision0", "test_recall0", "test_f1_score0", "test_support0",
             "test_precision1", "test_recall1", "test_f1_score1", "test_support1"])

        self.add_csv_entry(row_values)

    def _get_metrics(self, model, x_values, y_values):
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import roc_auc_score
        precision, recall, f1_score, support = precision_recall_fscore_support(y_values, model.predict(x_values))
        roc_auc = roc_auc_score(y_values, model.score(x_values))
        youden = recall[0] + recall[1] - 1
        return youden, roc_auc, precision[0], recall[0], f1_score[0], support[0], precision[1], recall[
            1], f1_score[1], support[1]
