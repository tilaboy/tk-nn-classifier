"""unit tests for classifier utils functions"""
import os
import tempfile
import json
import filecmp
import shutil
from unittest import TestCase
from tk_nn_classifier.data_loader.label_class_mapper import LabelClassMapper

class LabelClassMapperTestCases(TestCase):
    """unit tests"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.gold_mapping_file = 'tests/resource/label_mapper.json'
        self.test_mapping_file = os.path.join(self.test_dir, 'label_mapping.json')

    def tearDown(self):
        """remove the temp dir when test finished"""
        shutil.rmtree(self.test_dir)

    def test_mapper_from_diction(self):
        cats_dict = {'1': 'foo', '2': 'bar', '3': 'zoo'}
        mapper = LabelClassMapper(cats_dict)

        self.assertEqual(
            mapper.label_to_classid,
            {'foo': '1', 'bar': '2', 'zoo': '3'}
        )

    def test_mapper_from_list(self):
        labels = ['foo', 'bar', 'zoo']
        mapper = LabelClassMapper.from_labels(labels)

        self.assertEqual(
            mapper.label_to_classid,
            {'foo': '1', 'bar': '0', 'zoo': '2'}
        )
        self.assertEqual(
            mapper.classid_to_label,
            {'1': 'foo', '0': 'bar', '2': 'zoo'}
        )

    def test_mapper_from_list(self):
        labels = ['foo', 'bar', 'zoo', 'foo', 'zoo', 'foo']
        mapper = LabelClassMapper.from_labels(labels)

        self.assertEqual(
            mapper.label_to_classid,
            {'foo': '1', 'bar': '0', 'zoo': '2'}
        )
        self.assertEqual(
            mapper.classid_to_label,
            {'1': 'foo', '0': 'bar', '2': 'zoo'}
        )

    def test_writer(self):
        id_to_label = {'1': 'foo', '0': 'bar', '2': 'zoo'}
        mapper = LabelClassMapper(id_to_label, self.test_mapping_file)
        mapper.write()
        self.assertTrue(filecmp.cmp(
            self.gold_mapping_file,
            self.test_mapping_file,
            shallow=False))

    def test_mapper_from_file(self):
        mapper = LabelClassMapper.from_file(self.gold_mapping_file)
        self.assertEqual(
            mapper.label_to_classid,
            {'foo': '1', 'bar': '0', 'zoo': '2'}
        )
        self.assertEqual(
            mapper.classid_to_label,
            {'1': 'foo', '0': 'bar', '2': 'zoo'}
        )

    def test_mapping(self):
        mapper = LabelClassMapper.from_file(self.gold_mapping_file)

        self.assertEqual( mapper.class_id('zoo'), '2')
        self.assertEqual( mapper.label_name('2'), 'zoo')

    def test_equal_mapper(self):
        mapper = LabelClassMapper.from_file(self.gold_mapping_file)
        other_mapper = LabelClassMapper.from_labels(
                ['foo', 'zoo', 'foo', 'zoo', 'foo', 'bar', 'bar'])
        print(mapper)
        print(other_mapper)
        self.assertEqual(mapper, other_mapper)
