"""
base_args is set on import with 'load_config_modules(base_args)' #
"""

MultiStepLR = dict(
    object="optim.lr_scheduler.MultiStepLR",
    milestones=[1, 5, 10, 15],
    gamma=0.4
)
