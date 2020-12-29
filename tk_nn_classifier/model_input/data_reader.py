import os

from .label_class_mapper import LabelClassMapper
from ..config import data_field_type, FEAT_FIELD, CAT_FIELD
from ..data_loader.data_utils import iter_flatten


class DataReader():
    def __init__(self, config):
        self.config = config
        if 'label_mapper' not in self.config['datasets']:
            self.config['datasets']['label_mapper'] = os.path.join(
                    self.config['model_path'],
                    'label_mapper.json'
            )

        if os.path.isfile(self.config['datasets']['label_mapper']):
            self.label_mapper = LabelClassMapper.from_file(
                self.config['datasets']['label_mapper'])
        else:
            self.label_mapper = None

    def is_feature_field(self, field):
        return data_field_type(field, self.config) == FEAT_FIELD

    def is_category_field(self, field):
        return data_field_type(field, self.config) == CAT_FIELD

    def _build_label_mapper(self, labels):
        if self.label_mapper is None:
            self.label_mapper = LabelClassMapper.from_labels(
                    labels,
                    self.config['datasets']['label_mapper']
            )
            self.label_mapper.write()
