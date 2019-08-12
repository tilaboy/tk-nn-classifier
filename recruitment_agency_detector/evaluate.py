import tensorflow as tf
from tensorflow import keras

from data_utils import DataReader
from recruitment_agency_detector import LOGGER

class Evaluater:
    def __init__(self, type):
        self.type = type

    def tf_evaluation(model, test_file, data_reader):
        """Evaluate on the data set"""

        text_lines, x_eval, y_eval = data_reader.read_file(test_file)
        results = model.predict(x_eval)
        for text, i in enumerate(text_lines):
            LOGGER.info("predicted={:0.2f}\tclass={}\ttext={}".format(
                results[i][0], y_eval[i], text))
