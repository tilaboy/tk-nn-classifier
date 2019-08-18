'''The classifiers in different packages'''

from .spacy_classifier import SpaceClassifier
from .tf_classifier import TFClassifier

__all__ = ['SpacyClassifier', 'TFClassifier']
name = 'classifiers'
