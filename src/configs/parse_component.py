import argparse
from configs.parse import add_group, add_arguments
from configs.module_loader import load_module_safe


def load_config_modules(base_args):
    modules = []
    for name in ["dataset", "optimizer", "lr_scheduler", "model", "criterion", "trainer"]:
        module_name = f"configs.train.defaults.{name}"
        module = load_module_safe(module_name, base_args=base_args)
        modules.append(module)
    return modules


def parse_base(base_cfg):
    base_parser = argparse.ArgumentParser(add_help=False)
    add_arguments(base_parser, base_cfg)

    base_args, _ = base_parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[base_parser])
    return base_args, parser


def compose_config(parser, base_args, dataset, optimizer, lr_scheduler, model, criterion, trainer):
    for name, cfg in [
        ("dataset", dataset),
        ("optimizer", optimizer),
        ("lr_scheduler", lr_scheduler),
        ("model", model),
        ("criterion", criterion),
        ("trainer", trainer)
    ]:
        add_group(parser, base_args, cfg, name)
