from .data_reader import DataReader


class TFDataReader(DataReader):

    def get_data(self, data_path):
        data_set = self.get_data_set(data_path)
        texts, labels = zip(*data_set)
        self._build_label_mapper(labels)
        cats = [
            int(self.label_mapper.class_id(label))
            for label in labels]
        
        return texts, cats
