import os

class ConfigError(Exception):

    def __init__(self, field, detail_msg=''):
        msg = f"\nMissing/Wrong '{field}' configuration, please check config file\n"
        msg += f"details: {detail_msg}"
        super(ConfigError, self).__init__(msg)


class ResourceError(Exception):
    pass
