import random
from .data_reader import DataReader

class SpacyDataReader(DataReader):
    def get_data(self, data_path, shuffle=False, train_mode=False):
        data_set = list(self._get_data_set(data_path))
        if shuffle:
            random.shuffle(data_set)
        texts, labels = zip(*data_set)
        self._build_label_mapper(labels)
        cats = self._prepare_label(labels)
        if train_mode:
            cats = self._wrap_training_categories(cats)
        return list(zip(texts, cats))

    def split_train_test_data(self, data_path):
        """prepare data from our dataset."""
        train_data = list(
            self.get_data(self.config['datasets']['all_data']))
        random.shuffle(train_data)

        texts, labels = zip(*train_data)
        self._build_label_mapper(labels)
        cats = self._prepare_label(labels)
        split = int(len(train_data) * self.config['datasets']['split_ratio'])

        train_set = list(zip(
                texts[:split],
                self._wrap_training_categories(cats[:split])
                ))
        eval_set = list(zip(texts[split:], cats[split:]))
        return (train_set, eval_set)

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
