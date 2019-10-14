import random
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

        # texts are array of input features given in the config
        texts = ['\n'.join(feature) for feature in features]
        self._build_label_mapper(labels)
        cats = self._prepare_label(labels)
        return texts, cats



    def split_train_test_data(self, data_path):
        """prepare data from our dataset."""
        data_set = self.get_data(self.config['datasets']['all_data'])
        texts, cats = self._unpack_data(data_set, shuffle=True)
        split = int(len(data_set) * self.config['datasets']['split_ratio'])

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
