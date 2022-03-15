import argparse
import contextlib

import easydict
import yaml


def list2dict(args):

    _tuples = []
    for a in args:

        k, v = a.split("=")
        if k.startswith("--"):
            k = k[2:]

        try:  # IF IT'S A NUMBER
            if v.isdigit():  # TRY INTEGER FIRST
                v = int(v)
            else:  # AND FLOAT HERE.
                v = float(v)  # ERROR WILL OCCUR IF NON-CASTABLE STRING GIVEN

        except:  # JUST LET v STRING
            pass

        _tuples.append((k, v))

    return {k: v for (k, v) in _tuples}


def parse_args():

    parser = argparse.ArgumentParser()
    _, unknownargs = parser.parse_known_args()

    return list2dict(unknownargs)


if __name__ == "__main__":

    args = parse_args()
    print(args)


CONFIG_FILE_PATH = "./config.yml"


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, easydict.EasyDict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def save_config(cfg, path=CONFIG_FILE_PATH):

    if isinstance(cfg, easydict.EasyDict):
        cfg = edict2dict(cfg)

    else:
        pass

    with open(path, "w") as y:
        yaml.dump(cfg, y)


def load_config(path=CONFIG_FILE_PATH):

    with open(path, "r") as y:
        return easydict.EasyDict(yaml.load(y, Loader=yaml.Loader))


import contextlib


@contextlib.contextmanager
def using_config(name, value, cfg=None):
    old_value = getattr(cfg, name)
    setattr(cfg, name, value)
    try:
        yield
    finally:
        setattr(cfg, name, old_value)
