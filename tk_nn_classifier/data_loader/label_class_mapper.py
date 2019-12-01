'''Label mapping: create mapping from label to class_id'''
import json
import os
from .. import LOGGER


class LabelClassMapper:

    def __init__(self, classid_to_label, label_mapper_file='label_mapper.json'):
        self.classid_to_label = classid_to_label
        self.label_to_classid = { label: str(class_id) for class_id, label in classid_to_label.items()}
        self.label_mapper_file = label_mapper_file

    @classmethod
    def from_labels(cls, labels, label_mapper_file='label_mapper.json'):
        classes_to_label = {
                str(i):label
                for i, label in enumerate(sorted(set(labels)))
        }
        return cls(classes_to_label, label_mapper_file)

    @classmethod
    def from_file(cls, label_mapper_file):
        with open(label_mapper_file, 'r') as l_fh:
            classid_to_label = json.load(l_fh)
        return cls(classid_to_label, label_mapper_file)

    def write(self):
        dir = os.path.dirname(self.label_mapper_file)

        if dir is not '':
            os.makedirs(dir, exist_ok=True)

        with open(self.label_mapper_file, 'w') as l_fh:
            json.dump( self.classid_to_label, l_fh )

    def class_id(self, label):
        return self.label_to_classid[label]

    def label_name(self, class_id):
        return self.classid_to_label[str(class_id)]

    def __repr__(self):
        return str(self.label_to_classid)

    def __eq__(self, other):
        return self.label_to_classid == other.label_to_classid
