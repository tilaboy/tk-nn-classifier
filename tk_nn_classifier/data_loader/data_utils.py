from typing import Iterable
import os


def iter_flatten(items:Iterable) -> Iterable:
    '''Yield items from any nested iterable'''
    for item in items:
        if isinstance(item, Iterable) and \
                not isinstance(item, (str, bytes)):
            for sub_item in iter_flatten(item):
                yield sub_item
        else:
            yield item


def file_ext(input_file):
    '''get the file type from the file extension'''
    _, file_extension = os.path.splitext(input_file)
    file_ext = file_extension[1:] if file_extension else ''
    return file_ext


def file_itt(data_file):
    '''read each line of the file, generate the line '''
    with open(data_file, mode="r", encoding="utf-8") as data_fh:
        for line in data_fh:
            if not line.isspace():
                yield line.rstrip("\n")
