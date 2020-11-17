''' CSV file reader: import data from csv files'''
import random
import os
import csv
from .. import LOGGER
from .common_loader import CommonLoader


class CSVLoader(CommonLoader):
    def _train_fields(self):
        return super()._get_train_fields('csv_fields')

    def _detail_fields(self):
        return super()._get_detail_fields('csv_fields')

    def get_train_data(self, data_path):
        return self._get_values_from_csv(self._train_fields(), data_path)

    def get_details(self, data_path):
        return self._get_values_from_csv(self._detail_fields(), data_path)

    def _get_values_from_csv(self, fields, data_path):
        # to skip some csv file with BOM <U+FEFF> in the beginning
        with open(data_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                yield [
                    row[field] if isinstance(field, str) else
                    [
                        self._prepare_input_text(row[sub_field], index == 0)
                        for sub_field in field
                    ]
                    for index, field in enumerate(fields)
                ]

    @staticmethod
    def _split_docs_on_ratio(data_path, ratio, random_shuffle=False):
        with open(data_path, newline='', encoding='utf-8-sig') as csvfile:
            rows = list(csv.reader(csvfile))
            header = rows.pop(0)
        if not rows:
            raise ValueError('no rows in %s, please check config' % data_path)
        if random_shuffle is True:
            random.shuffle(rows)
        split_point = int(len(rows) * ratio)
        train_rows = rows[:split_point]
        eval_rows = rows[split_point:]

        LOGGER.info('split %d records into %d train and %d eval',
                    len(rows),
                    len(train_rows),
                    len(eval_rows)
                    )

        return header, train_rows, eval_rows

    def split_data(self, data_path, ratio=0.8, des='models'):
        '''split the data into train and evel'''
        header, train_rows, eval_rows = self._split_docs_on_ratio(
            data_path, ratio, random_shuffle=True)

        if des:
            os.makedirs(des, exist_ok=True)
            train_file = os.path.join(des, 'train.csv')
            eval_file = os.path.join(des, 'eval.csv')
            LOGGER.info('write the train data to train file %s' % train_file)
            LOGGER.info('write the eval data to eval file %s' % eval_file)

        else:
            raise ValueError('train/eval destination needs to be specified')

        with open(train_file, 'w', newline='', encoding='utf-8') as train_fh:
            csv_writer = csv.writer(train_fh, delimiter=",",
                                    quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header)
            csv_writer.writerows(train_rows)

        with open(eval_file, 'w', newline='', encoding='utf-8') as eval_fh:
            csv_writer = csv.writer(eval_fh, delimiter=",",
                                    quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header)
            csv_writer.writerows(eval_rows)

        return train_file, eval_file
