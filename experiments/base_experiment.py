"""
Everything pertaining to all experiments
"""
from abc import ABC, abstractmethod
import sys
import datetime
import logging
import settings


class Experiment(ABC):
    """
    Provides a standard base for experiment conducting.
    """

    def __init__(self, log_file_location=None):
        self._log_file_location = log_file_location
        self.timestamp = datetime.datetime.now()
        self.has_run = False
        self.had_error_on_run = None
        self.logger = None
        self.is_running = False

    @abstractmethod
    def _experiment(self):
        """
        Write code for the experiment in this method.
        :return: None
        """
        pass

    @property
    def log_file_location(self):
        """
        Gets the name of the log file associated with an instance of this
        Experiment
        :return: The location of the log file as a string.
        """
        if self._log_file_location is not None:
            return self._log_file_location
        return settings.DEFAULT_LOG_FOLDER + self.name + ".log"

    def run(self):
        """
        The method used to run the experiment. The usage in main.py should
        usually be of the format:
            ExperimentSubclass().run()
        with _experiment() never explicitly called.
        This method logs the information in a custom file.
        :return: None
        """
        self._initialize_logger()
        self.logger.info('Experiment Name: %s', self.name)
        self.logger.info('DateTime: %s', datetime.datetime.now())
        with open("settings.py", "r") as settings_file:
            self.logger.info('Contents of \'settings.py\':\n%s', settings_file.read())
        self.logger.info('Starting Experiment')
        start_time = datetime.datetime.now()

        try:
            self.is_running = True
            self._experiment()
            self.had_error_on_run = False
        except Exception:
            all_general_exceptions = sys.exc_info()[0]
            self.logger.error(str(all_general_exceptions), exc_info=True)
            self.had_error_on_run = True
        finally:
            self.is_running = False
            end_time = datetime.datetime.now()
            self.logger.info('Experiment ended.')
            self.logger.info('Total Time Elapsed: %s', str(end_time-start_time))
            self.has_run = True

    def _initialize_logger(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        # Create a file and console handler
        file_handler = logging.FileHandler(self.log_file_location)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # Create the most common type of formatter and add it to the handlers
        format_string = self.name + ': %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_string)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    @property
    def name(self):
        """
        Get the name of the experiment.
        :return: The name of the experiment.
        """
        datetime_string = self.timestamp.strftime("%Y-%m-%d--%H-%M-%S")
        name = datetime_string + "---" + str(self.__class__.__name__)
        return name

    def log(self, message):
        """
        Shortcut method for logging default messages.
        :param message: The string to log.
        :return: The output of the logger.
        """
        return self.logger.info(message)

    def log_error(self, error_message):
        """
        Shortcut method for logging error messages.
        :param error_message: The string to log.
        :return: The output of the logger.
        """
        return self.logger.error(error_message)

    def add_csv_entry(self, list_of_entries, filename=None):
        """
        Appends a line to a CSV file
        :param list_of_entries: Each entry in list_of_entries will be added to the CSV file.
        :param filename: The name of the CSV file.
        :return: None
        """
        try:
            if filename is None:
                filename = settings.DEFAULT_RESULTS_FOLDER + self.name + ".csv"
            with open(filename, "a") as csv_file:
                entry_count = len(list_of_entries)
                for index, entry in enumerate(list_of_entries):
                    if index + 1 == entry_count:
                        csv_file.write(str(entry) + "\n")
                    else:
                        csv_file.write(str(entry) + ",")
        except:
            self.log_error("Error Writing Entry to CSV File")

