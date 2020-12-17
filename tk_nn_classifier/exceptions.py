import os

class ConfigError(Exception):
    def __init__(self, field, section='', detail_msg=''):
        msg = f'\nMissing/Wrong "{field}" value'
        if section:
            msg += f' in "{section}"'
        msg += ', please check config file.\n'
        if detail_msg:
            msg += f'details: {detail_msg}'
        super().__init__(msg)


class ResourceError(Exception):
    pass


class FileTypeError(Exception):
    def __init__(self, file_type):
        msg = f'\nfile type "{file_type}" not supported.\n'
        super().__init__(msg)
