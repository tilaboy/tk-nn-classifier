import random
from shutil import copyfile
import os
from os import listdir
import csv
from xml_miner.miner import TRXMLMiner
from collections import Iterable
from .label_class_mapper import LabelClassMapper
from .common_data_reader import CommonDataReader

class TRXMLLoader(CommonDataReader):
    def _train_fields(self):
        return super()._get_train_fields('trxml_fields')

    def _detail_fields(self):
        return super()._get_detail_fields('trxml_fields')

    def get_train_data(self, data_path):
        return self._get_values_from_trxml(self._train_fields(), data_path)

    def get_details(self, data_path):
        return self._get_values_from_trxml(self._detail_fields(), data_path)

    def _get_values_from_trxml(self, fields, data_path):
        # the first element in the fields is the input text to the data models
        trxml_miner = TRXMLMiner(','.join(list(self._iter_flatten(fields))))
        for trxml in trxml_miner.mine(data_path):
            yield [
                trxml['values'][field] if isinstance(field, str) else
                [
                    self._prepare_input_text(trxml['values'][sub_field], index==0)
                    for sub_field in field
                ]
                for index, field in enumerate(fields)
            ]

    @staticmethod
    def split_data_on_ratio(
                   data_path,
                   ratio=0.8,
                   des='models',
                   random_shuffle=False
                  ):
        files = listdir(data_path)
        if not files:
            raise ValueError('no file found in %s, please check config' % data_path)
        if random_shuffle == True:
            random.shuffle(data_set)
        split_point = int(len(files) * ratio)
        train_files = files[:split_point]
        eval_files = files[split_point:]

        return train_files, eval_files


    def backup_splitted_data(self)
        train_files, eval_files = self.split_data_on_ratio()

        #copyfile()


        # TODO copy them to the new des
