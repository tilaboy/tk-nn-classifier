from . import LOGGER
from .classifiers import TFClassifier, SpaceClassifier

class Model:
    def __init__(self, config):
        self.config = config
        self.type = config['model_type']
        if self.type == 'tf':
            self.classifier = TFClassifier(config)
        elif self.type == 'spacy':
            self.classifier = SpaceClassifier(config)
        else:
            raise ValueError("unknown classifier type [{}]".format(self.type))

    def build_graph(self):
        self.classifier.build_graph()

    def train(self, train_data, eval_data):
        self.classifier.train(train_data, eval_data)

    def save(self, output_path):
        self.classifier.save(output_path)

    def build_and_train(self):
        self.classifier.build_and_train()

    def evaluate(self, test_data_path):
        self.classifier.evaluate(test_data_path)
