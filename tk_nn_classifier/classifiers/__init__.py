'''The classifiers in different packages'''

from .spacy_classifier import SpacyClassifier
from .tf_classifier import TFClassifier
from .keras_classifier import KerasClassifier

__all__ = ['SpacyClassifier', 'TFClassifier', 'KerasClassifier']
name = 'classifiers'
