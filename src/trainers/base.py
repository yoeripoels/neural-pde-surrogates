import argparse
import timeit
import warnings
from abc import abstractmethod, ABCMeta
from typing import *

import torch
import os
from torch import optim
from torch.utils.data import DataLoader

from utils import misc as util
from common.interfaces import M, D
from data.base import DatasetInterface
from models.base import ModelInterface
from utils.collate_batch_helpers import collate_data
WANDB_AVAILABLE = True
try:
    import wandb
except ModuleNotFoundError:
    WANDB_AVAILABLE = False


class TrainInterface(metaclass=ABCMeta):
    def __init__(
        self,
        model: ModelInterface,
        data: DatasetInterface,
        criterion: Callable,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler,
        config: argparse.Namespace = None,
        save_path: str = "models/model.pt",
        max_train_batches=float("inf"),
        max_test_batches=float("inf"),
        epoch_callback: callable = None,
        use_wandb=False,
        wandb_kwargs=None,
        wandb_config_dict=None,
        **kwargs,
    ):

        self.model = model
        self.data = data
        if config is not None:
            self.config = config
        else:
            self.config = argparse.Namespace(**kwargs)
        self.config.save_path = save_path
        if self.data.data_interface in [D.sim1d_var_t]:
            self.config.variable_time = True  # make sure we treat it as variable time
        elif not hasattr(self.config, 'variable_time'):
            self.config.variable_time = False  # if not detected / set otherwise, assume False

        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.max_train_batches = max_train_batches
        self.max_test_batches = max_test_batches

        self.epoch_callback = epoch_callback
        if not hasattr(self.config, "print_setting"):
            self.print_setting = dict(print_per_step=False)
        else:
            self.print_setting = self.config.print_setting


        # WandB check:
        if WANDB_AVAILABLE:
            self.use_wandb = use_wandb
        else:  # dont use wandb because we don't want to or it is not available
            self.use_wandb = False
            if use_wandb:
                warnings.warn('Could not import WandB -- WandB not used!')
        self.wandb_kwargs = wandb_kwargs
        self.wand_config_dict = wandb_config_dict

        if hasattr(self.config, "test_kwargs_list"):
            self.test_kwargs_list = self.config.test_kwargs_list
        else:
            self.test_kwargs_list = [('default', {})]

    def __repr__(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def model_interface(self) -> List[M]:
        """A list for storing what model interfaces the training method is compatible with"""
        raise NotImplementedError("model_interface not set!")

    @property
    @abstractmethod
    def data_interface(self) -> List[D]:
        """A list for storing what data interfaces the training method is compatible with"""
        raise NotImplementedError("data_interface not set!")

    def get_parameters(self):
        return self.model.parameters()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def train_step(
        self, batch: Tuple, epoch, batch_idx, loader,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Method that should be implemented for a TrainInterface class. This method is called by the train method of the
        TrainInterface class that has already been implemented. Only the forward pass's loss calculation
        should be implemented; the backward pass and learning step is taken care of outside this method.
        Args:
            batch: tuple with a batch of data -- you should implement here how to get model input and labels from the data.
                    This will depend on the type of training (e.g. temporal bundling, pushforward, ...) and model
                    (graph, grid, ...), and what kind of data is available. Typically this is the channels, spatial
                    coordinates and possibly PDE parameters. If the batch tuple contains tensors, these have been moved
                     to the specified device already for the forward pass.
            epoch: index of epoch
            batch_idx: index of batch
            loader: the DataLoader used for training
        Returns:
            Zero-dimensional tensor with loss value for this batch, predictions for this batch
        """
        raise NotImplementedError("The method train_step should be implemented!")
        return loss_of_this_batch, preds

    def test_step(
        self, batch: Tuple, batch_idx: int, use_train_loss_calc=False, include_data=False, **kwargs
    ) -> Tuple[Union[torch.Tensor, float], dict]:
        """
        To be implemented for a specific testing strategy
        Args:
            batch: batch with data
            batch_idx: index of the batch
            use_train_loss_calc: boolean indicating if train_step is to be used - will be set to True if test_step
                raises a NotImplementedError.
            include_data: boolean indicating whether to include the ground truth + simulated data in the output dict
        """
        if include_data:
            raise ValueError("include_data is only supported when implemented in test_step")
        if not use_train_loss_calc:
            raise NotImplementedError("The test_step method is not implemented!")
        else:
            loss, preds = self.train_step(batch, epoch=0, batch_idx=batch_idx)

        # when implementing test_step you can save other statistics than the primary loss in a dict and return it
        dict_with_some_statistics = {}

        return loss, dict_with_some_statistics

    def __call__(self):
        self.train()

    def get_dataloaders(self):
        """
        Helper function to retrieve dataloaders based on the specified config.

        Returns:
            DataLoader train data
            DataLoader valid data
            DataLoader test data
        """
        persistent_workers = self.config.nw > 0
        pin_memory = True
        if not self.config.variable_time:
            dataloader_kwargs = dict(
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.nw,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            )

            train_loader = DataLoader(self.data.train, **dataloader_kwargs)
            valid_loader = DataLoader(self.data.valid, **dataloader_kwargs)
            test_loader = DataLoader(self.data.test, **dataloader_kwargs)
        else:
            sampler_obj = self.config.sampler["object"]
            sampler_kwargs = dict(self.config.sampler)
            del sampler_kwargs["object"]

            collate_fn_min = collate_data(mode='min', return_lengths=True, tw=self.config.time_window)
            collate_fn_max = collate_data(mode='max', return_lengths=True, tw=self.config.time_window)

            sampler_train = sampler_obj(self.data.train, batch_size=self.config.batch_size, **sampler_kwargs)
            if "with_replacement" in sampler_kwargs:
                del sampler_kwargs["with_replacement"]
            sampler_valid = sampler_obj(self.data.valid, batch_size=self.config.batch_size, with_replacement=False, **sampler_kwargs)
            sampler_test = sampler_obj(self.data.test, batch_size=self.config.batch_size, with_replacement=False, **sampler_kwargs)

            dataloader_kwargs = dict(
                num_workers=self.config.nw,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            )
            train_loader = DataLoader(
                self.data.train,
                batch_sampler=sampler_train,
                collate_fn=collate_fn_min,
                **dataloader_kwargs,
            )
            valid_loader = DataLoader(
                self.data.valid,
                batch_sampler=sampler_valid,
                collate_fn=collate_fn_max,
                **dataloader_kwargs,
            )
            test_loader = DataLoader(
                self.data.test,
                batch_sampler=sampler_test,
                collate_fn=collate_fn_max,
                **dataloader_kwargs,
            )
        return train_loader, valid_loader, test_loader

    def train(
        self,
    ) -> Tuple[
        List[Union[float, torch.Tensor]], Dict[str, List[Union[float, torch.Tensor]]], Dict[str, List[Dict]]
    ]:
        """
        Trains the model with which the class was initialized on the dataset with which it was initialized
        Args:

        Returns:
            list of training losses per epoch, list of val losses per epoch, list of (dict of val stats) per epoch
        """

        # check that the model/data interfaces fit
        assert (
            self.model.model_interface in self.model_interface
        ),  f"{self} does not support model {self.model}."  # model must be supported by training method
        assert (
            self.data.data_interface in self.model.data_interface
        ), f"{self.model} does not support data from {self.data}."  # data must be supported by model
        assert (
            self.data.data_interface in self.data_interface
        ),  f"{self} does not support data from {self.data}."  # data method must be supported by training method

        util.check_directory()

        train_loader, valid_loader, test_loader = self.get_dataloaders()

        # WandB init if we use it:
        if self.use_wandb:
            wandb.init(config=self.wand_config_dict, **self.wandb_kwargs)
            # wandb.watch(self.model)
        # Training loop

        # Will try test step first and fall back to train_step for testing if test_step is not implemented
        fall_back_to_training_loss_calc = False

        train_losses = []
        min_val_loss = {name: float("inf") for (name, _) in self.test_kwargs_list}
        val_losses = {name: [] for (name, _) in self.test_kwargs_list}
        val_stats_list = {name: [] for (name, _) in self.test_kwargs_list}
        time_start = timeit.default_timer()
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            train_losses.append(train_loss)

            if (epoch + 1) % self.config.print_interval == 0:
                time_total = timeit.default_timer() - time_start
                # compute the progress towards the next validation step (as we don't necessarily do this every epoch)
                if (epoch + 1) % self.config.test_interval == 0:
                    progress = 1.0
                else:
                    epoch_next = epoch + 1
                    epoch_prev_test = (
                        epoch_next - epoch_next % self.config.test_interval
                    )
                    progress = (epoch + 1 - epoch_prev_test) / self.config.test_interval
                print(
                    f"Epoch {epoch} (progress: {progress:.2f}, {time_total:.4f}s), Loss {train_loss}"
                )
                time_start = timeit.default_timer()

            # convenient to have all loggings for all test_kwargs of the latest epoch in one dict, rather than
            # per name in the val_stats_list dict. we also send this object to WandB logger if applicable
            this_epoch_logging_dict = {'train_loss': train_loss}
            if (epoch + 1) % self.config.test_interval == 0:
                for name, test_kwargs in self.test_kwargs_list:
                    print(f"Evaluation on validation dataset for setting [{name}]:")
                    if isinstance(test_kwargs, Callable):
                        with torch.no_grad():
                            val_loss, val_stats = test_kwargs(valid_loader, self)
                    else:
                        try:
                            val_loss, val_stats = self.test(
                                valid_loader, fall_back_to_training_loss_calc, test_kwargs=test_kwargs,
                            )
                        except NotImplementedError:
                            warnings.warn(
                                "test_step method not implemented."
                                "Falling back to training loss calculation for validation set performance!"
                            )
                            fall_back_to_training_loss_calc = True
                            val_loss, val_stats = self.test(
                                valid_loader, fall_back_to_training_loss_calc, test_kwargs=test_kwargs,
                            )
                    print(f"Evaluation metric: {util.to_float(val_loss)}")
                    if not self.print_setting["print_per_step"]:
                        val_stats = {k: v for (k, v) in val_stats.items() if 'step' not in k.lower()}
                    val_stats_floatdict = util.to_floatdict(val_stats)
                    print(util.dict_str(val_stats_floatdict, prefix='-'))
                    print()

                    # also append the log of val stats in the last epoch for all test kwargs:
                    this_epoch_logging_dict[name + ' - val loss'] = val_loss
                    for k, v in val_stats_floatdict.items():
                        this_epoch_logging_dict[name + '-' + str(k)] = v

                    # logging per name-test kwargs over all epochs
                    val_losses[name].append(val_loss)
                    val_stats_list[name].append(val_stats)
                    if val_loss < min_val_loss[name]:
                        # Save model
                        self.save_model(self.config.save_path + f"_{name}.pt")
                        min_val_loss[name] = val_loss

                        # Evaluate on test set
                        print("Found new best model, evaluation on test dataset:")
                        if isinstance(test_kwargs, Callable):
                            with torch.no_grad():
                                test_loss, test_stats = test_kwargs(test_loader, self)
                        else:
                            test_loss, test_stats = self.test(
                                test_loader, fall_back_to_training_loss_calc, test_kwargs=test_kwargs,
                            )
                        print(f"Test metric: {util.to_float(test_loss)}")
                        if not self.print_setting["print_per_step"]:
                            test_stats = {k: v for (k, v) in test_stats.items() if 'step' not in k.lower()}
                        print(util.dict_str(util.to_floatdict(test_stats), prefix='-'))
                        print()

            # WandB logging if applicable:
            if self.use_wandb:
                wandb.log(this_epoch_logging_dict)

        # save final model
        self.save_model(self.config.save_path + f"_final.pt")
        if self.use_wandb:
            wandb.finish()
        return train_losses, val_losses, val_stats_list

    def save_model(self, save_name):
        filename, ext = os.path.splitext(save_name)
        if ext == '':  # default extension
            ext = '.pt'
        save_name = filename + ext
        torch.save(self.model.state_dict(), save_name)
        print(f"Saved model at {save_name}")

    def simulate(self, u, *args, compute_loss=True, include_data=True, nr_gt_steps=1, t_res=100, **kwargs
                 ) -> Union[torch.Tensor,
                            list,
                            Tuple[torch.Tensor, Tuple[list, list]]]:
        """
        Simulate using specified u for the initial condition (and loss/misc, if indicated)
        Args:
            u: Simulation data
            *args: Misc. simulation args
            compute_loss: Whether to compute the loss using u (and in case of include_data, also
                          return the ground-truth data)
            include_data: Whether to include the (simulated) data
            nr_gt_steps: Number of ground-truth steps the solver is being ran for
            t_res: Number of steps to simulate for
            **kwargs: Misc. simulation kwargs

        Returns:

        """
        raise NotImplementedError("The method simulate is not implemented!")

    def test(
        self, loader: torch.utils.data.DataLoader, use_train_loss_calc=False, include_data=False, test_kwargs=None,
    ) -> Union[Tuple[Union[float, torch.Tensor], dict],
               Tuple[Union[float, torch.Tensor], dict, Tuple[torch.Tensor, list]]]:
        """
        Test a model on a dataset. Calls train_one_epoch function but without tracking gradients
        Args:
            loader: dataloader on which to be tested
        Returns:
            float with loss
        """
        if test_kwargs is None:
            test_kwargs = {}

        if hasattr(loader, "batch_sampler"):
            if loader.batch_sampler.batch_size != self.config.batch_size:
                print("Alert: batch_size in the supplied dataloader sampler is not equal to that in the config.")
        elif loader.batch_size != self.config.batch_size:
            print("Alert: batch_size in the supplied dataloader is not equal to that in the config.")

        device = self.config.device
        self.model.eval()
        loss = 0
        other_metrics = {}
        n_total = 0
        if include_data:
            data_gt, data_pred, data_other = [], [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch_on_device = tuple(
                    t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
                )
                # we reuse the train_step method for testing, without keeping track of the gradients.
                # however, we now can keep track of multiple metrics (all that were passed in the criteria dict)

                test_step_out = self.test_step(
                    batch_on_device, batch_idx, use_train_loss_calc, include_data, **test_kwargs
                )
                if include_data:
                    batch_loss, batch_other_metrics, batch_data = test_step_out
                else:
                    batch_loss, batch_other_metrics = test_step_out

                # get the number of elements in this batch
                batch_size = util.get_batch_size(batch)

                loss += batch_loss * batch_size
                n_total += batch_size

                for k, v in batch_other_metrics.items():
                    if k in other_metrics:
                        other_metrics[k] += v * batch_size
                    else:
                        other_metrics[k] = v * batch_size
                if include_data:
                    data_gt.append(batch_data[0])
                    data_pred.append(batch_data[1])
                    data_other.extend(batch_data[2])

                if batch_idx >= self.max_test_batches - 1:
                    break
        loss = loss / n_total
        other_metrics = {k: v / n_total for k, v in other_metrics.items()}
        if include_data:
            data_gt = [x.to('cpu') for x in data_gt]  # handle final data processing on CPU
            data_pred = [x.to('cpu') for x in data_pred]
            if self.data.data_interface in [D.sim1d_var_t]:
                # get max t
                max_t = max([x.shape[2] for x in data_gt])
                # pad data as needed
                data_gt_pad, data_pred_pad = [], []
                for gt, pred in zip(data_gt, data_pred):
                    t_size = gt.shape[2]
                    if t_size == max_t:  # if t matches, simply append
                        data_gt_pad.append(gt)
                        data_pred_pad.append(pred)
                    else:  # otherwise, pad with 0s
                        data_shape = list(gt.shape)  # create our pad-shape
                        data_shape[2] = max_t
                        gt_pad = torch.zeros(data_shape)
                        gt_pad[:, :, :t_size] = gt
                        pred_pad = torch.zeros(data_shape)
                        pred_pad[:, :, :t_size] = pred
                        data_gt_pad.append(gt_pad)
                        data_pred_pad.append(pred_pad)
                data_gt = data_gt_pad
                data_pred = data_pred_pad

            data_gt = torch.cat(data_gt, dim=0)
            data_pred = torch.cat(data_pred, dim=0)
            return loss, other_metrics, (torch.stack([data_gt, data_pred], dim=0), data_other)
        else:
            return loss, other_metrics

    def train_one_epoch(self, loader: torch.utils.data.DataLoader, epoch) -> float:
        """
        method that loops over all batches for training in an epoch.
        Args:
            loader: Pytorch Dataloader that returns x, labels tensors for train_batch method
            epoch: index of epoch
        Returns:
            float of loss value for this epoch
        """
        self.model.train()
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        device = self.config.device
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            batch_on_device = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
            )
            optimizer.zero_grad()
            loss, preds = self.train_step(batch_on_device, epoch, batch_idx, loader=loader)
            loss.backward()
            optimizer.step()
            # get the number of elements in this batch
            batch_size = util.get_batch_size(batch)
            total_loss += loss.detach() / batch_size

            if batch_idx >= self.max_train_batches:
                break
        total_loss = total_loss / len(loader)
        if self.epoch_callback is not None:
            self.epoch_callback(self, loader, epoch)

        if lr_scheduler is not None:
            if (epoch + 1) % self.config.lr_step_interval == 0:
                lr_scheduler.step()
        return total_loss
