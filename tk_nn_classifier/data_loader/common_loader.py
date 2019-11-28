''' Basic class to read files also get the field names relevant to the training'''
from collections import Iterable
from .. import LOGGER

class CommonLoader:
    def __init__ (self, config):
        self.max_lines = config['max_lines']
        self.config = config

    def _prepare_input_text(self, text, to_clean=False):
        if to_clean:
            lines = text.split("\n")
            text = "\n".join(lines[:self.max_lines])
        return text

    def _iter_flatten(self, items):
        """Yield items from any nested iterable; see Reference."""
        for item in items:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                for sub_item in self._iter_flatten(item):
                    yield sub_item
            else:
                yield item

    def get_train_data(self, data_path):
        raise NotImplementedError('get_train_data needs to be implemented')

    def get_details(self, data_path):
        raise NotImplementedError('get_details needs to be implemented')

    def _get_train_fields(self, cfg_entry):
        fields = [self.config[cfg_entry]['features'],
                  self.config[cfg_entry]['class']]
        return fields

    def _get_detail_fields(self, cfg_entry):
        fields = [self.config[cfg_entry]['features'],
                  self.config[cfg_entry]['class'],
                  self.config[cfg_entry]['doc_id']]
        if 'extra' in self.config[cfg_entry]:
            fields += self.config[cfg_entry]['extra']
        return fields
