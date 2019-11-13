'''The classifiers in different packages'''

from .spacy_classifier import SpaceClassifier
from .tf_classifier import TFClassifier
from .tf_multi_feat_classifier import TFMultiFeatClassifier

__all__ = ['SpacyClassifier', 'TFClassifier', 'TFMultiFeatClassifier']
name = 'classifiers'
