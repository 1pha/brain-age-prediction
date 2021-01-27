import yaml

CONFIG_FILE_PATH = './config.yml'

def write_config(cfg, changes: dict, path=CONFIG_FILE_PATH):

    for key, item in changes.items():
        cfg[key] = item

    with open(path, 'w+') as yml_config_file:
        yaml.dump(cfg.get_dict(), yml_config_file, default_flow_style=False)


def load_config(path=CONFIG_FILE_PATH):

    with open(path, 'r') as yml_config_file:
        return CFG(yaml.load(yml_config_file))


class CFG:
    def __init__(self, cfg):
        self.cfg_dict = cfg
        for c in cfg:
            setattr(self, c, cfg[c])

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.cfg_dict[key] = value
        return setattr(self, key, value)

    def refresh(self):
        for key in self.cfg_dict:
            self.cfg_dict[key] = getattr(self, key)

    def get_dict(self):
        return self.cfg_dict

    def keys(self):
        return self.cfg_dict.keys()

    def items(self):
        return self.cfg_dict.items()