import yaml

def load_config(args):
    base_config_path = f'merg_code/config/base.yaml'
    with open(base_config_path) as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    args.update(base_config)
    return args
    