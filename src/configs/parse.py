import importlib
import types
import sys
import argparse
from argparse import Namespace
from configs.module_loader import load_module_safe
import random
from copy import copy
import ast


import utils


class ConfigGroupArg:
    def __init__(self, v):
        self.v = v

    def __call__(self):
        return self.v


def add_arguments(parser, cfg):
    for k, v in cfg.items():
        parser.add_argument(f"--" + k, type=parse_arg_default(type(v)), default=v)


def get_custom_group_titles(parser):
    default_groups = ['positional arguments', 'optional arguments']
    return [group.title for group in parser._action_groups if group.title not in default_groups]


def add_group(parser, base_args, cfg, group_name):
    group = parser.add_argument_group(group_name)
    parser.add_argument(  # important that we add the group argument first (for parsing order)
        f"--{group_name}",
        type=str,
        default=None,
        help=None
    )
    if isinstance(cfg, dict):
        iterator_values = cfg.items()
        group_prefix = lambda a, _: f'--{group_name}.{a}'
    elif isinstance(cfg, list) or isinstance(cfg, tuple):
        iterator_values = enumerate(cfg)
        group_prefix = lambda a, _: f'--{group_name}[{a}]'
    else:
        raise ValueError("'cfg' must be dict, list or tuple")

    for k, v in iterator_values:
        help = None
        if k in base_args:
            help = argparse.SUPPRESS
        if isinstance(v, dict) and len(v) > 0:  # always unroll dicts
            if isinstance(cfg, dict):
                add_group(parser, base_args, v, f"{group_name}.{k}")
            elif isinstance(cfg, tuple) or isinstance(cfg, list):
                add_group(parser, base_args, v, f"{group_name}[{k}]")
        elif (isinstance(v, tuple) or isinstance(v, list)) and \
                any([(isinstance(x, dict) or isinstance(x, tuple) or isinstance(x, list)) for x in v]):
            # if we have any dicts/lists/tuples in our list/tuple, unroll
            if isinstance(cfg, dict):
                add_group(parser, base_args, v, f"{group_name}.{k}")
            elif isinstance(cfg, tuple) or isinstance(cfg, list):
                add_group(parser, base_args, v, f"{group_name}[{k}]")
        else:
            group.add_argument(
                group_prefix(k, v),
                type=parse_arg_default(type(v)),
                default=ConfigGroupArg(v),
                help=help,
            )


def flatten(d, parent_key="", sep="."):
    items = []
    if isinstance(d, types.SimpleNamespace) or isinstance(d, argparse.Namespace):
        d = d.__dict__
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if (
            isinstance(v, dict)
            or isinstance(v, types.SimpleNamespace)
            or isinstance(v, argparse.Namespace)
        ):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_defaults(key, value, base_args):
    module_name = f'configs.train.defaults.{key}'
    module = load_module_safe(module_name, base_args=base_args)
    try:
        value_dict = getattr(module, value)
    except AttributeError:
        raise Exception(f"Error loading default for '--{key}': '{value}' cannot be found in '{module_name}'")
    return value_dict


def parse_list(part):
    """
    Parse str and return idx, if -1, it is not a list, e.g.
    part={name} returns {name, -1}
    part={name[5]} return {name, 5}
    """
    if part.endswith("]"):  # we are unflattening a list, not a dict
        part_idx = int(part[part.index("[") + 1:part.index("]")])
        part_name = part[:part.index("[")]  # get before the bracket
        return part_name, part_idx
    else:
        return part, -1


def parse_boolean(value):
    """
    Simple function to parse str input to a bool
    """
    value = str(value).lower()
    if value in ['true', 't', 'yes', 'y', '1']:
        return True
    elif value in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


def unflatten(d: dict, parser, sep="."):
    assert isinstance(d, dict)
    result = dict()

    argument_groups = get_custom_group_titles(parser)
    argument_groups_override = set()
    base_args = None
    base_args_parsed = False

    for key, value in d.items():
        if key in argument_groups:
            if not base_args_parsed:  # if we're at the first argument group, all base args are parsed --> save
                base_args_parsed = True
                base_args = Namespace(**dict(result))
            if value is not None:  # load from defaults if specified
                result[key] = get_defaults(key, value, base_args)
                argument_groups_override.add(key)
            continue  # do not do regular parsing either way for argument group arguments

        if isinstance(value, ConfigGroupArg):  # check if we're loading from config or argv
            value = value()
            config_arg = True
        else:
            config_arg = False
        parts = key.split(sep)

        parts_new = []  # handle [x] appearing
        for p in parts:
            if p.count(']') > 1:
                p_out = [x + ']' for x in p.split(']')[:-1]]
            else:
                p_out = [p]
            parts_new.extend(p_out)
        parts = parts_new

        argument_group = parts[0]

        if argument_group in argument_groups_override and config_arg:
            # if we have overwritten this group, don't re-load items of the group from the config
            # however, we do want to parse cmd arguments still for the items of the group (hence the config_arg check)
            continue

        d = result
        for part in parts[:-1]:
            part, idx = parse_list(part)

            if idx == -1:  # is a dict
                if part not in d:
                    d[part] = dict()
                d = d[part]
            else:  # idx >= 0
                if part not in d:
                    d[part] = dict(__is_list__=True)
                if idx not in d[part]:
                    d[part][idx] = dict()
                d = d[part][idx]
        part, idx = parse_list(parts[-1])
        if idx == -1:  # is value
            d[parts[-1]] = value
        else:  # idx > 0:
            if part not in d:
                d[part] = dict(__is_list__=True)
            if idx not in d[part]:
                d[part][idx] = dict()
            d[part][idx] = value

    # remove dummy dicts
    result = remove_dummy_dict(result, dummy='')
    # post-process -> flatten dicts with __is_list__=True to list
    result = flatten_dict_to_list(result, "__is_list__", True)

    return result


def remove_dummy_dict(d, dummy):
    if not isinstance(d, dict):
        return d

    if len(d) == 1 and dummy in d:  # found dummy, fix
        return list(d.values())[0]
    else:  # recursively fix dict
        return {k: remove_dummy_dict(v, dummy) for k, v in d.items()}


def flatten_dict_to_list(d, key, value):
    """
    If key in d and d[key]==value, this dictionary represents a list and will be flattened
    """
    if not isinstance(d, dict):  # reached bottom of recursion, return
        return d

    if key in d and d[key] == value:  # found match, flatten
        d.pop(key)
        # assert that our keys are 0, 1, 2...N
        assert sorted(d.keys()) == list(range(len(d))), "dictionary contains non-index keys"
        d_list = []
        for i in range(len(d)):
            d_list.append(flatten_dict_to_list(d[i], key, value))
        return d_list

    # no need to flatten; recursively flatten elements
    d_out = {}
    for k, v in d.items():
        d_out[k] = flatten_dict_to_list(v, key, value)
    return d_out


def parse_arg_default(default_type):
    def parse_arg(arg):
        # check if list
        if str(arg)[0] == "[" and str(arg)[-1] == "]":
            return ast.literal_eval(arg)
        elif isinstance(None, default_type):  # default_type is the type of the default arg, so in case of None
            return str(arg)  # parse the new input as str
        elif default_type == type(True):  # check if we are parsing a bool
            return parse_boolean(arg)
        else:
            return default_type(arg)
    return parse_arg


def get_config_from_sys_argv():
    argv = sys.argv
    if "--config" in argv:
        config_file = argv[argv.index("--config") + 1]
        index = argv.index("--config")
    elif "-C" in argv:
        config_file = argv[argv.index("-C") + 1]
        index = argv.index("-C")
    else:
        raise Exception("No config file specified (use -C or --config).")

    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    return config_file


def parse_cfg():
    config_file = get_config_from_sys_argv()

    try:
        cfg = copy(
            importlib.import_module(
                config_file.replace("/", ".").replace(".py", "")
            ).cfg
        )
    except ModuleNotFoundError:
        raise Exception(f"Config file {config_file} not found.")

    # This enables all arguments to be shown in the help menu.
    parser = argparse.ArgumentParser()
    all_args = flatten(cfg)

    for k, v in all_args.items():
        if k in all_args:
            parser.add_argument("--" + k, type=parse_arg_default(type(v)), default=v)

    args = parser.parse_args()

    for k, v in vars(args).items():
        if v != utils.rgetattr(cfg, k):
            print(f"Updating {k} to {v}")
            utils.rsetattr(cfg, k, v)

    if args.seed > -1:
        cfg.seed = args.seed

    if cfg.seed < 0:
        cfg.seed = random.randint(0, sys.maxsize)

    print("Seed: ", cfg.seed)

    utils.set_seed(cfg.seed)
    return cfg


def parse_args(config_file=None):
    if config_file is None:
        config_file = get_config_from_sys_argv()
    try:
        parser = copy(
            importlib.import_module(
                config_file.replace("/", ".").replace(".py", "")
            ).parser
        )
    except (ModuleNotFoundError, AttributeError):
        raise Exception(f"Cannot access 'parser' attribute of {config_file}")
    args = parser.parse_args()
    args = unflatten(vars(args), parser)
    print(f"Loaded config: {config_file} {' '.join(sys.argv[1:])}")
    utils.set_seed(args["seed"])
    print(f"Set seed to {args['seed']}")
    return args
