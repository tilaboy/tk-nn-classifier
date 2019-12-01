'''SpaCy data reader: prepare the train/eval data in spaCy format'''
import random
from .. import LOGGER
from .data_reader import DataReader


class SpacyDataReader(DataReader):
    def get_data(self, data_path, shuffle=False, train_mode=False):
        data_set = self.get_data_set(data_path)
        texts, cats = self._unpack_data(data_set, shuffle)
        if train_mode:
            cats = self._wrap_training_categories(cats)
        return list(zip(texts, cats))

    def _unpack_data(self, data_set, shuffle=False):
        if shuffle:
            random.shuffle(data_set)
        features, labels = zip(*data_set)

        texts = [
            feature if isinstance(feature, str) else '\n'.join(feature)
            for feature in features]
        self._build_label_mapper(labels)
        cats = self._prepare_label(labels)
        return texts, cats

    def _prepare_label(self, labels):
         return [
            {
                class_type: class_type == label
                for class_type in self.label_mapper.label_to_classid
            }
            for label in labels
        ]

    @staticmethod
    def _wrap_training_categories(cats):
        return [{"cats": cat} for cat in cats]
