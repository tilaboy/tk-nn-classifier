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

    def train(self, train_data=None, eval_data=None):
        if train_data:
            self.classifier.train(train_data, eval_data)
        else:
            self.classifier.train()

    def save(self, output_path):
        self.classifier.save(output_path)

    def load(self, model_dir):
        self.classifier.load_model()

    def build_and_train(self):
        self.classifier.build_and_train()

    def evaluate(self, test_data_path):
        self.classifier.evaluate(test_data_path)
