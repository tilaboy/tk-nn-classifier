from .data_reader import DataReader


class TFDataReader(DataReader):
    def get_data(self, data_path, config):
        data_set = list(self._get_data_set(data_path))
        texts, labels = zip(*data_set)
        self._build_label_mapper(labels)
        cats = [
            self.label_mapper.label_to_classid(label)
            for label in labels]
        return list(zip(texts, cats))
