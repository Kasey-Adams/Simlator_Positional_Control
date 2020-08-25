import numpy as np


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    # __getattr__ = dict.get # does not raise KeyError if key missing
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        return self[key]


def readparamfile(filename, params=None):
    if params is None:
        params = dotdict({})
    with open(filename) as file:
        for line in file.readlines():
            line, _, _ = line.partition('#')  # remove comments
            trimmedline = line.translate(str.maketrans('', '', ' \n'))
            if trimmedline:
                key, _, value = trimmedline.partition('=')
                value = eval(value)

                # elif value[0].isnumeric():
                #     if '.' in value or 'e' in value:
                #         value = float(value)
                #     else:
                #         value = int(value)
                # elif value=='True' or value=='False':
                #     value = bool(value)
                # elif value[0:5] == 'eval(':
                #     value = eval(value)
                # elif value[0] in ("'", '"'):
                #     value = value[1:-1]
                # else:
                #     pass # key left as string
                params[key] = value
    return params
