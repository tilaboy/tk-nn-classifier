'''The classifiers in different packages'''

from .spacy_classifier import SpacyClassifier
from .tf_classifier import TFClassifier
from .keras_classifier import KerasClassifier
from .tf_multi_feat_classifier import TFMultiFeatClassifier

__all__ = ['SpacyClassifier', 'TFClassifier', 'KerasClassifier', 'TFMultiFeatClassifier']
name = 'classifiers'
