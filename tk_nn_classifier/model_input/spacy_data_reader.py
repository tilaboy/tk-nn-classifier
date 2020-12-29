'''SpaCy data reader: prepare the train/eval data in spaCy format'''
import random
from .data_reader import DataReader


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
        features, _, labels, _ = self.get_feature_values(data_gen)
        texts = ['\n'.join(feature) for feature in features]
        self._build_label_mapper(labels)
        cats = self._prepare_label(labels)
        return texts, cats

    def get_feature_values(self, data_gen):
        feature_fields = list()
        category_field = None
        first_doc = next(data_gen)
        for field in first_doc.keys():
            if self.is_feature_field(field):
                feature_fields.append(field)
            elif self.is_category_field(field):
                category_field = field
        features = [[first_doc[field] for field in feature_fields]]
        labels = [first_doc[category_field]]

        for doc in data_gen:
            features.append([doc[field] for field in feature_fields])
            labels.append(doc[category_field])
        return features, feature_fields, labels, category_field

    def _prepare_label(self, labels):
        return [
            {category: category == label
             for category in self.label_mapper.label_to_classid}
            for label in labels]

    @staticmethod
    def _wrap_training_categories(cats):
        return [{"cats": cat} for cat in cats]
