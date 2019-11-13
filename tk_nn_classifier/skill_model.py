'''Use the skill-validation Neural Network model'''

from tensorflow import keras
from data_loader import DataReader

class SkillModel:
    """Dedicated class for inference"""

    def __init__(self, embeddings_model_file, nn_model_file):
        """
        Load the word embeddings and the NN classification model.
        Set up the tensorflow session config.
        """

        self.embeddings_reader = DataReader(
            embeddings_model_file,
            token_encoding="max_embedding",
            data_format="service_mode")

        self.classifier = keras.models.load_model(nn_model_file)

    def predict_skill_likelihood(self, match):
        """
        Predict if given match example is a valid skill or not.
        Expect object with left_context, surface_form, right_context attributes
        """

        encoded_input = self.embeddings_reader.read_match(match)
        return float(self.classifier.predict_on_batch(encoded_input)[0])

    def predict_skills_likelihood(self, matches):
        """
        Predict if given match examples are valid skills or not.
        Expect objects with left_context, surface_form, right_context attributes
        """

        encoded_input = self.embeddings_reader.read_matches(matches)
        predictions = self.classifier.predict_on_batch(encoded_input)
        return list(predictions.flatten())
