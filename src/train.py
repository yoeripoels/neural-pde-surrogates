from argparse import Namespace
import torch

import os
import tempfile
import sys
from datetime import datetime
import pickle

from configs.parse import parse_args
from utils import rgetattr, count_parameters
from utils import misc as util


PRINT_ARGS = True

import data
import models
import trainers


def get_config_static(args, lazy_init=True, model_override=None):
    # load device
    device = args["trainer"]["device"]

    # initialize dataset object
    # NOTE: the pde object (containing dt, tmin, tmax, etc.) is created here, always stored in dataset.pde
    dataset = getattr(data, args["dataset"].pop("object"))(**args["dataset"])

    # initialize model
    model_name = args["model"]["object"]  # save for later use
    model = getattr(models, args["model"].pop("object"))(**args["model"], pde=dataset.pde).to(device)
    if model_override is not None:
        model = model_override
    # initialize criterion
    criterion = rgetattr(torch, args["criterion"].pop("object"))(**args["criterion"])

    # set up save path
    config = Namespace(**args["trainer"])
    if args["experiment_name"] is None:
        dateTimeObj = datetime.now()
        timestring = f"{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}{dateTimeObj.time().microsecond}"
        # args["experiment_name"] = f"{dataset.pde}_{dataset}_{model_name}_res{config.base_resolution[1]}-{config.super_resolution[1]}_tw{config.time_window}_time{timestring}"
        args["experiment_name"] = f"{dataset.pde}_{model_name}_{timestring}"

    save_path = os.path.join(args["experiment_path"], args["experiment_name"])

    use_wandb = args['use_wandb']
    wandb_kwargs = args['wandb_kwargs']
    if use_wandb:
        wandb_config_dict = {}
        for k, v in args.items():
            if not isinstance(v, dict):
                wandb_config_dict[k] = v
            else:
                wandb_config_dict[k] = dict(v)
    else:
        wandb_config_dict = {}

    # load remaining trainer arguments
    if "epoch_callback" in args["trainer"]:
        epoch_callback = args["trainer"]["epoch_callback"]
        args["trainer"].pop("epoch_callback")
    else:
        epoch_callback = None

    optimizer = None
    lr_scheduler = None
    # initialize trainer object, the backbone of all usage
    trainer = getattr(trainers, args["trainer"].pop("object"))(
        model=model,
        data=dataset,
        config=config,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_path=save_path,
        epoch_callback=epoch_callback,
        use_wandb=use_wandb,
        wandb_kwargs=wandb_kwargs,
        wandb_config_dict=wandb_config_dict
    )

    # optimizer and lr_scheduler are optionally set -->
    # in case we do lazy initialization of layers, we want to set them after a forward pass of the model has been done
    # if they are not lazily initialized, we can safely initialize and set them here
    if not lazy_init:
        optimizer = rgetattr(torch, args["optimizer"].pop("object"))(
            trainer.get_parameters(), **args["optimizer"]
        )
        trainer.set_optimizer(optimizer)
        try:
            lr_scheduler = rgetattr(torch, args["lr_scheduler"].pop("object"))(
                optimizer, **args["lr_scheduler"]
            )
        except KeyError:
            lr_scheduler = None
        trainer.set_lr_scheduler(lr_scheduler)
    return device, dataset, model_name, model, criterion, trainer, optimizer, lr_scheduler


def main(args):
    # overwrite stdout w/ class that does .flush() after each print statement (nice for slurm)
    default_stdout = sys.stdout
    sys.stdout = util.Logger(default_stdout, write_log=False)

    # torch.autograd.set_detect_anomaly(True)
    if PRINT_ARGS:
        print(util.dict_str(args, prefix='--', mapping='='))

    experiment_path = args["experiment_path"]
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # allow for lazy initialization of model params, so don't set optimizer/lr_scheduler here
    device, dataset, model_name, model, criterion, trainer, _, _ = \
        get_config_static(args, lazy_init=True)

    print(f"Save path set to {trainer.config.save_path}")
    # start run: print device information
    if torch.cuda.is_available() and device.lower() != "cpu":
        device_name = " (" + torch.cuda.get_device_name(torch.cuda.current_device()) + ")"
    else:
        device_name = ""
    print(f"Loaded device: {device}" + device_name)
    # pre-train check on valid data
    print("Sanity check on validation data...")
    _, valid_loader, test_loader = trainer.get_dataloaders()
    print('shape of one datapoint: (bs, channels, time, *spatial)', next(iter(valid_loader))[1].size())
    valid_loss, valid_summary = trainer.test(valid_loader)
    print("Pre-train valid summary:")
    print(util.dict_str(util.to_floatdict(valid_summary), prefix=' • '))

    # set optimizer / lr_scheduler after forward pass, in case of lazy initialization
    optimizer = rgetattr(torch, args["optimizer"].pop("object"))(
        trainer.get_parameters(), **args["optimizer"]
    )
    try:
        lr_scheduler = rgetattr(torch, args["lr_scheduler"].pop("object"))(
            optimizer, **args["lr_scheduler"]
        )
    except KeyError:
        lr_scheduler = None
    trainer.set_optimizer(optimizer)
    trainer.set_lr_scheduler(lr_scheduler)

    if args["function_pre"] is not None:
        print("\n\nRunning pre-training callable")
        args["function_pre"](args, dataset, optimizer, lr_scheduler, model, criterion, trainer)

    print('\n\n----Start training----')
    print("Number of parameters:", count_parameters(trainer.get_parameters(), provided_as_params=True))

    train_losses, val_losses, val_stats = trainer.train()
    print("Train losses:", util.to_floatlist(train_losses))
    print("Validation losses:", util.to_floatlist(val_losses[list(val_losses.keys())[0]]))

    with open(os.path.join(args["experiment_path"], args["experiment_name"] + "_train_summary.pickle"), "wb") as f:
        pickle.dump(dict(train_losses=train_losses, val_losses=val_losses, val_stats=val_stats), f)

    # do a final run on test data
    test_loss, test_summary = trainer.test(test_loader)
    print("Test loss:", util.to_float(test_loss))
    print("Test summary:")
    print(util.dict_str(util.to_floatdict(test_summary), prefix=' • '))

    if args["function_post"] is not None:
        print("\n\nRunning post-training callable")
        args["function_post"](args, dataset, optimizer, lr_scheduler, model, criterion, trainer)

    print("Run Completed!")


if __name__ == "__main__":
    args = parse_args()

    exception = None
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            main(args)
        except (Exception, KeyboardInterrupt) as e:
            exception = e

    if exception is None:
        print("Run finished!")
    else:
        raise (exception)
