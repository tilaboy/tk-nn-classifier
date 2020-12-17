'''SpaCy data reader: prepare the train/eval data in spaCy format'''
import random
from .data_reader import DataReader
from ..config import DATA_LABEL_FIELD


class SpacyDataReader(DataReader):
    def model_input(self, data_gen, train_mode):
        texts, cats = self._unpack_data(data_gen)
        if train_mode:
            cats = self._wrap_training_categories(cats)
        return list(zip(texts, cats))

    def _unpack_data(self, data_gen):
        '''
        converting the data_set to model input

        features, e.g. three text features
        [[f1_a, f1_b, f1_c],
         [f2_a, f2_b, f2_c],
         .....
        ]

        labels:
        [l1, l2, l3, .....]
        '''
        features, labels = self.get_feature_values(data_gen)
        texts = ['\n'.join(feature) for feature in features]
        self._build_label_mapper(labels)
        cats = self._prepare_label(labels)
        return texts, cats

    def get_feature_values(self, data_gen):
        features = list()
        labels = list()
        feature_fields = list()
        category_field = None
        for doc in data_gen:
            if category_field is None:
                for field in doc.keys():
                    if self.is_feature_field(field):
                        feature_fields.append(field)
                    elif self.is_category_field(field):
                        category_field = field
            feature = [doc[field] for field in feature_fields]
            features.append(feature)
            labels.append(doc[category_field])
        return features, labels

    def _prepare_label(self, labels):
        return [
            {class_type: class_type == label
             for class_type in self.label_mapper.label_to_classid}
            for label in labels]

    @staticmethod
    def _wrap_training_categories(cats):
        return [{"cats": cat} for cat in cats]
