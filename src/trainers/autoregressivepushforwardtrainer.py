import random
from typing import *

import torch
import math
from utils import misc as util

from common.data_creator import DataCreator
from common.interfaces import D, M
from trainers.base import TrainInterface
from utils.collate_batch_helpers import create_data_mask
from utils.process_output import process_step


class AutoregressivePushforwardTrainer(TrainInterface):
    # set supported interface
    data_interface = [D.sim1d, D.sim2d, D.sim1d_var_t]
    model_interface = [M.AR_TB, M.AR_TB_GNN]

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
        **kwargs
        )

        config = self.config
        # if we use a GNN model, the below class takes care of processing the data into graph format
        data_creator = DataCreator(
            pde=self.data.pde,
            neighbors=config.neighbors if hasattr(config, "neighbors") else 3,
            time_window=config.time_window,
            t_resolution=config.base_resolution[0],
            x_resolution=config.base_resolution[1],
        ).to(config.device)

        if not hasattr(self.config, "process_settings"):
            self.config.process_settings = {}

        self.data_creator = data_creator  # not used if we use a non-graph model

    def train_step(self, batch: Tuple, epoch, batch_idx, loader):
        """
        Implements pushforward training for autoregressive models.

        Extended from implementation in https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers

        Args:
            batch: tuple with a batch of data -- you should implement here how to get model input and labels from the data.
                    This will depend on the type of training (e.g. temporal bundling, pushforward, ...) and model
                    (graph, grid, ...), and what kind of data is available. Typically this is the channels, spatial
                    coordinates and possibly PDE parameters. If the batch tuple contains tensors, these have been moved
                     to the specified device already for the forward pass.
            epoch: Id of the epoch
            batch_idx: Id of the batch within the epoch
            loader: The dataloader reference that was used to provide the batch
        Returns:
            tuple of (Zero-dimensional tensor with loss value for this batch, model output for this batch)
        """

        # the config class needs to have the below properties:
        batch_size = self.config.batch_size
        device = self.config.device
        criterion = self.criterion

        if self.data.data_interface == D.sim1d_var_t:
            u_base, u_super, x, conditioning, t_conditioning, spatial_conditioning, batch_lengths = batch
            t_res = u_super.shape[2]
        else:
            u_base, u_super, x, conditioning, t_conditioning, spatial_conditioning = batch
            t_res = self.data_creator.t_res
        use_t_conditioning = torch.numel(t_conditioning) != 0
        if torch.numel(spatial_conditioning) == 0:
            spatial_conditioning = None

        # for now, assume we want to increase unrolling evenly with increasing the lr step
        if self.data.data_interface not in [D.sim1d_var_t]:  # static time simulation
            unrolling_epoch = epoch // self.config.lr_step_interval
            max_unrolling = min(unrolling_epoch, self.config.unrolling)
            unrolling = list(range(max_unrolling + 1))
            unrolled_graphs = random.choice(unrolling)
        else:
            # get # of unrolled graphs from sampler in the varying time case
            unrolled_graphs = loader.batch_sampler.get_t_batch(batch_idx)

        steps = [
            t
            for t in range(
                self.data_creator.tw,
                t_res - self.data_creator.tw - (self.data_creator.tw * unrolled_graphs) + 1,
            )
        ]

        random_steps = random.choices(steps, k=batch_size)
        data, labels = self.data_creator.create_data(u_super, random_steps)
        if self.model.model_interface == M.AR_TB_GNN:
            graph = self.data_creator.create_graph(
                data, labels, x, conditioning, random_steps
            ).to(device)
        elif self.model.model_interface == M.AR_TB:
            data, labels = data.to(device), labels.to(device)

        # get BCs for first step
        _, bc = process_step(pde=self.data.pde, model_interface=self.model.model_interface, sim_prev=data, gtbc_prev=data,
                             gtbc_next=labels, device=device, get_bc=True, set_bc=False, set_min=False,
                             process_settings=self.config.process_settings)

        # get cond for first step
        if use_t_conditioning:
            t_cond = self.data_creator.create_data(t_conditioning, random_steps, mode="labels")
        else:
            t_cond = None

        with torch.no_grad():
            for _ in range(unrolled_graphs):
                # perform model prediction
                if self.model.model_interface == M.AR_TB_GNN:
                    pred = self.model(graph, cond=conditioning, bc=bc, pos=x, t_cond=t_cond, spatial_cond=spatial_conditioning)
                elif self.model.model_interface == M.AR_TB:
                    data = self.model(data, cond=conditioning, bc=bc, pos=x, t_cond=t_cond, spatial_cond=spatial_conditioning)

                # ---- create data for next iteration: ----
                labels_prev = labels  # state at time t+1 - what we're trying to predict with the model in this iter.
                random_steps = [rs + self.data_creator.tw for rs in random_steps]
                # get labels at time t+2 - target for next iteration:
                _, labels = self.data_creator.create_data(u_super, random_steps)

                if self.model.model_interface == M.AR_TB_GNN:
                    graph = self.data_creator.create_next_graph(
                        graph, pred, labels, random_steps
                    ).to(device)
                elif self.model.model_interface == M.AR_TB:
                    labels = labels.to(device)

                # postprocess data -- set BCs/min values in data, extract new BC info
                data, bc = process_step(pde=self.data.pde, model_interface=self.model.model_interface, sim_prev=data,
                                        gtbc_prev=labels_prev, gtbc_next=labels, device=device, get_bc=True, set_bc=True,
                                        set_min=True, process_settings=self.config.process_settings)
                # get conditioning for next step
                if use_t_conditioning:
                    t_cond = self.data_creator.create_data(t_conditioning, random_steps, mode="labels")
                else:
                    t_cond = None

        # get prediction
        if self.model.model_interface == M.AR_TB_GNN:
            pred = self.model(graph, cond=conditioning, bc=bc, pos=x, t_cond=t_cond, spatial_cond=spatial_conditioning)
        elif self.model.model_interface == M.AR_TB:
            pred = self.model(data, cond=conditioning, bc=bc, pos=x, t_cond=t_cond, spatial_cond=spatial_conditioning)

        # set BCs / min of prediction
        pred, _ = process_step(pde=self.data.pde, model_interface=self.model.model_interface, sim_prev=pred,
                               gtbc_prev=labels, gtbc_next=None, device=device, get_bc=False, set_bc=True,
                               set_min=True, process_settings=self.config.process_settings)

        # handle loss
        if self.model.model_interface == M.AR_TB_GNN:
            loss = criterion(pred, graph.y)
        elif self.model.model_interface == M.AR_TB:
            loss = criterion(pred, labels)
        loss = torch.sqrt(loss)
        return loss, pred

    def test_step(
        self, batch: Tuple, batch_idx: int, use_train_loss_calc=False, include_data=False,
            max_test_len=None,
    ) -> Union[Tuple[Union[torch.Tensor, float], dict],
               Tuple[Union[torch.Tensor, float], dict, list]]:
        """
        uses unrolled predictions for calculating the primary validation loss. Additionally, returns base losses by
        ground-truth solver on lower-dimensional grid, unrolled fw losses, mean per-step loss and losses per step.
        Args:
            batch: batch with data
            batch_idx: index of the batch
            use_train_loss_calc: boolean indicating if train_step is to be used - will automatically
             be set to True if test_step raises a NotImplementedError. As such, if this function is not implemented,
             the loss for validation will be calculated in the same way as the training loss.
            include_data: boolean indicating whether to return data of all ground truth & prediction samples
            max_test_len: for variable-time setting, max length of evaluation
        Returns:
            loss tensor or float that is used as 'main validation loss' for model selection, dict with other validation
            performance metrics.
        """

        if use_train_loss_calc:
            msg = "We should probably not have use_train_loss=True when having implemented the test_step method..."
            raise RuntimeError(msg)

        # format of the data that we expect:
        if self.data.data_interface == D.sim1d_var_t:
            u_base, u_super, x, conditioning, t_conditioning, spatial_conditioning, batch_lengths = batch
            if max_test_len is not None:
                t_res = min(max_test_len, u_super.shape[2])
                batch_lengths = [t_res for _ in batch_lengths]
            else:
                t_res = u_super.shape[2]
            u_super_mask = create_data_mask(u_super, batch_lengths)
            use_mask = True
        else:
            u_base, u_super, x, conditioning, t_conditioning, spatial_conditioning = batch
            t_res = self.data_creator.t_res
            use_mask = False

        batch_size = u_super.shape[0]
        use_t_conditioning = torch.numel(t_conditioning) != 0

        # u_base: baseline solver solutions
        # u_super: high res solver solutions downsampled to lower resolution
        # x: spatial coordinates tensor
        # conditioning: pde variables
        # t_conditioning: time changing pde variables / BCs
        # spatial_conditioning: static spatial conditioning (e.g. spatial BCs)
        # (opt) batch_lengths: length of each simulation in batch (for padding)

        # first we check the losses for different timesteps (one forward prediction array!)
        steps = [
            t
            for t in range(
                self.data_creator.tw,
                t_res - self.data_creator.tw + 1,
                self.data_creator.tw
            )
        ]

        losses = []
        loss_step_dict = {}
        for step in steps:
            same_steps = [step] * batch_size
            data, labels = self.data_creator.create_data(u_super, same_steps)
            # get BCs
            _, bc = process_step(pde=self.data.pde, model_interface=self.model.model_interface, sim_prev=data,
                                 gtbc_prev=data, gtbc_next=labels, device=self.config.device, get_bc=True, set_bc=False,
                                 set_min=False, process_settings=self.config.process_settings)
            if use_mask:
                labels_mask = self.data_creator.create_data(u_super_mask, same_steps, mode="labels")
            if use_t_conditioning:
                t_cond = self.data_creator.create_data(t_conditioning, same_steps, mode="labels")
            else:
                t_cond = None
            if self.model.model_interface == M.AR_TB_GNN:  # GNN - use data_creator to process data into graph format
                #DEPRECATED
                graph = self.data_creator.create_graph(
                    data, labels, x, conditioning, same_steps
                ).to(self.config.device)
                pred = self.model(graph, cond=conditioning, bc=bc, pos=x, t_cond=t_cond)  # call model
            elif self.model.model_interface == M.AR_TB:  # no GNN, just keep tensor format
                data, labels = data.to(self.config.device), labels.to(
                    self.config.device
                )
                pred = self.model(data, cond=conditioning, bc=bc, pos=x,
                                  t_cond=t_cond, spatial_cond=spatial_conditioning)  # call model
                if use_mask:
                    labels_mask = labels_mask.to(self.config.device)
                    pred *= labels_mask
                    labels *= labels_mask
            # set BCs / min
            pred, _ = process_step(pde=self.data.pde, model_interface=self.model.model_interface, sim_prev=pred,
                                   gtbc_prev=labels, gtbc_next=None, device=self.config.device, get_bc=False, set_bc=True,
                                   set_min=True, process_settings=self.config.process_settings)
            if self.model.model_interface == M.AR_TB_GNN:
                loss = self.criterion(pred, graph.y)
            elif self.model.model_interface == M.AR_TB:
                loss = self.criterion(pred, labels)
            losses.append(loss / batch_size)
            loss_step_dict[f"Step {step}, mean loss"] = loss / batch_size

        losses = torch.stack(losses)
        # next we test the unrolled losses
        unrolled_losses_out = self._test_unrolled_losses(batch, include_data, max_test_len, divide_by_t=True)
        if include_data:
            unrolled_losses, unrolled_base_losses, data_simulations = unrolled_losses_out
        else:
            unrolled_losses, unrolled_base_losses = unrolled_losses_out

        out_info_dict = {
            "Unrolled base losses": unrolled_base_losses,
            "Unrolled forward losses": unrolled_losses,
            "Mean per-step loss": torch.mean(losses),
            **loss_step_dict,
        }

        if include_data:
            return torch.mean(unrolled_losses), out_info_dict, data_simulations
        else:
            return torch.mean(unrolled_losses), out_info_dict

    def simulate(self, u, conditioning, x, compute_loss, include_data, nr_gt_steps, t_res, t_conditioning=torch.empty(0),
                 spatial_conditioning=torch.empty(0), clip_min=True, use_bc=True, u_bc=None, u_mask=None, divide_by_t=True):
        """

        Args:
            u: The simulation data to use. Of shape [batch_size, channel, time, num_x]. Time must be at least
            the size of time window
            conditioning: Variables related to the simulations, dict of tensors
            x: The grid of the simulation(s)
            compute_loss: Whether to compute the loss and return groundtruth simulations
            include_data: Whether to return the simulated data
            nr_gt_steps: How many ground truth steps before using the model to predict
            t_res: Time for the full simulation
            t_conditioning (optional): Conditioning data to use, of shape [batch_size, channel, time]
            bc_u (optional): The simulation data to take BCs from. Assumed to be the same shape as u -->
            [batch_size, channel, time, num_x]; However, num_x need not be the same. We only take the values at the
            boundary, so 0 or -1, so the internal dimensions can be discarded.

            If set to None, u will be used.
            u_mask: If provided, outputs will be masked during the loss computation (useful during e.g. padding)
        Returns:
            unrolled_losses                           if compute_loss=True and include_data=False
            pred_data                                 if compute_loss=False and include_data=True
            unrolled_losses, (gt_data, pred_data)     if compute_loss=True and include_data=True
        """
        use_mask = u_mask is not None  # keep track of whether we want to use a mask
        if compute_loss is False and use_mask:
            raise ValueError("Mask supplied for computing the loss, but 'compute_loss'=False!")
        if compute_loss is True and u.shape[2] < t_res:
            raise ValueError("Cannot compute loss if no ground-truth simulation is provided for the full rollout")
        if u_bc is None:  # if no specific bc_info is provided
            u_bc = u  # extract the BC info from the ground-truth simulation data itself
        if use_bc and u_bc.shape[2] < t_res:
            raise ValueError("Cannot set BCs if the provided BC information is <= the unrolling time")
        if u.shape[2] < nr_gt_steps * self.data_creator.tw:
            raise ValueError(f"The training data is shorter than the specified number of unrolling steps: "
                             f"{nr_gt_steps} * {self.data_creator.tw} = {nr_gt_steps * self.data_creator.tw}"
                             f", but u.shape[2] = {u.shape[2]}")
        use_t_conditioning = torch.numel(t_conditioning) != 0  # whether to extract & use conditioning
        use_spatial_conditioning = torch.numel(spatial_conditioning) != 0  # whether to extract & use conditioning

        batch_size = u.shape[0]
        device = self.config.device

        same_steps = [self.data_creator.tw * nr_gt_steps] * batch_size
        pred = self.data_creator.create_data(u, same_steps, mode="data")  # get first input
        pred = pred.to(device)  # called pred as we use the name pred in for loop (i.e., they are predictions there)
        if use_bc:
            bc_cur = self.data_creator.create_data(u_bc, same_steps, mode="data")  # and first bc
            bc_cur = bc_cur.to(device)
        if self.model.model_interface == M.AR_TB_GNN:  # GNN needs graph variable, set to None
            graph = None

        if include_data:
            if compute_loss:
                data_gt = [pred]
            data_pred = [pred]

        if compute_loss:
            losses = []

        if use_mask and self.model.model_interface == M.AR_TB_GNN:
            print("ALERT: use_mask = True, but masking is currently not supported for GNN architectures!")

        # Unroll trajectory and add losses which are obtained for each unrolling
        n_t = 0
        for step in range(
                self.data_creator.tw * (nr_gt_steps),
                t_res - self.data_creator.tw + 1,
                self.data_creator.tw,
        ):
            same_steps = [step] * batch_size  # get step info

            # get info for computing loss
            if compute_loss:
                labels = self.data_creator.create_data(u, same_steps, mode="labels")
                labels = labels.to(device)
            if use_mask:
                labels_mask = self.data_creator.create_data(u_mask, same_steps, mode="labels")
                labels_mask = labels_mask.to(device)

            # get info for BCs
            if use_bc:
                bc_prev = bc_cur
                bc_cur = self.data_creator.create_data(u_bc, same_steps, mode="labels")
                _, bc = process_step(pde=self.data.pde, model_interface=self.model.model_interface, sim_prev=pred,
                                     gtbc_prev=bc_prev, gtbc_next=bc_cur, device=self.config.device, get_bc=True,
                                     set_bc=False, set_min=False, process_settings=self.config.process_settings)
            else:
                bc = None
            # get conditioning at next step
            if use_t_conditioning:
                t_cond = self.data_creator.create_data(t_conditioning, same_steps, mode="labels")
            else:
                t_cond = None

            if use_spatial_conditioning:
                spatial_cond = spatial_conditioning
            else:
                spatial_cond = None

            # get prediction
            if self.model.model_interface == M.AR_TB_GNN:
                if graph is None:  # on first iteration, initialize the graph
                    graph_labels = labels if compute_loss else pred
                    graph = self.data_creator.create_graph(
                        pred, graph_labels, x, conditioning, same_steps
                    ).to(device)
                else:  # otherwise, only update it
                    graph_labels = labels if compute_loss else pred
                    graph = self.data_creator.create_next_graph(graph, pred, graph_labels, same_steps).to(device)
                pred = self.model(graph, cond=conditioning, bc=bc, pos=x, t_cond=t_cond, spatial_cond=spatial_cond)
            elif self.model.model_interface == M.AR_TB:
                pred = self.model(pred, cond=conditioning, bc=bc, pos=x, t_cond=t_cond, spatial_cond=spatial_cond)
                if compute_loss and use_mask:
                    labels_mask = labels_mask.to(self.config.device)
                    pred *= labels_mask
                    labels *= labels_mask

            # set BCs/min
            if use_bc:
                pred, _ = process_step(pde=self.data.pde, model_interface=self.model.model_interface, sim_prev=pred,
                                       gtbc_prev=bc_cur, gtbc_next=None, device=self.config.device, get_bc=False,
                                       set_bc=True, set_min=False, process_settings=self.config.process_settings)
            if clip_min:
                pred, _ = process_step(pde=self.data.pde, model_interface=self.model.model_interface, sim_prev=pred,
                                       gtbc_prev=None, gtbc_next=None, device=self.config.device, get_bc=False,
                                       set_bc=False, set_min=True, process_settings=self.config.process_settings)

            # handle loss
            if compute_loss:
                if self.model.model_interface == M.AR_TB_GNN:
                    loss = self.criterion(pred, graph.y) / math.prod(self.config.base_resolution[1:])
                elif self.model.model_interface == M.AR_TB:
                    loss = self.criterion(pred, labels) / math.prod(self.config.base_resolution[1:])
                losses.append(loss / batch_size)
            if include_data:
                if compute_loss:
                    data_gt.append(labels)
                if self.model.model_interface == M.AR_TB_GNN:
                    pred_grid = torch.tensor(util.grid_graph_to_array(pred, graph.pos, graph.batch, self.data.pde.dxs)).to(self.config.device)
                    data_pred.append(pred_grid)
                elif self.model.model_interface == M.AR_TB:
                    data_pred.append(pred)
            n_t += self.data_creator.tw
        if divide_by_t:
            losses = [v / n_t for v in losses]
        if compute_loss and not include_data:
            return losses
        elif not compute_loss and include_data:
            return data_pred
        else:
            return losses, (data_gt, data_pred)

    def _test_unrolled_losses(self, batch, include_data=False, max_test_len=None, divide_by_t=True) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                                        Tuple[torch.Tensor, torch.Tensor, list]]:
        """
        Helper method for calculating loss for full trajectory unrolling.
        Args:
            batch: batch with data
            include_data: boolean indicating whether we return the simulations
        Returns:
            torch.Tensor: unrolled loss
        """

        if self.data.data_interface == D.sim1d_var_t:
            u_base, u_super, x, conditioning, t_conditioning, batch_lengths = batch
            if max_test_len is not None:
                t_res = min(max_test_len, u_super.shape[2])
                batch_lengths = [t_res for _ in batch_lengths]
            else:
                t_res = u_super.shape[2]
            u_super_mask = create_data_mask(u_super, batch_lengths)
        else:
            u_base, u_super, x, conditioning, t_conditioning, spatial_conditioning = batch
            t_res = self.data_creator.t_res
            u_super_mask = None

        unroll_out = self.simulate(u_super, conditioning, x, t_conditioning=t_conditioning, spatial_conditioning=spatial_conditioning,
                                   compute_loss=True, include_data=include_data,
                                   nr_gt_steps=self.config.nr_gt_steps, t_res=t_res, u_mask=u_super_mask, divide_by_t=divide_by_t)
        if include_data:
            losses_tmp, (data_gt, data_pred) = unroll_out
        else:
            losses_tmp = unroll_out

        batch_size = u_super.shape[0]

        losses_base_tmp = []
        # Losses for numerical baseline
        n_t = 0
        for step in range(
                self.data_creator.tw * self.config.nr_gt_steps,
                t_res - self.data_creator.tw + 1,
            self.data_creator.tw,
        ):
            if torch.numel(u_base) == 0:
                losses_base_tmp.append(torch.Tensor(0))
                continue
            same_steps = [step] * batch_size
            _, labels_super = self.data_creator.create_data(u_super, same_steps)
            _, labels_base = self.data_creator.create_data(u_base, same_steps)

            loss_base = (
                self.criterion(labels_super, labels_base)
                / math.prod(self.config.base_resolution[1:])
            )
            losses_base_tmp.append(loss_base / batch_size)
            n_t += self.data_creator.tw

        # post process data
        if include_data:
            data_gt = torch.cat(data_gt, dim=2)
            data_pred = torch.cat(data_pred, dim=2)
            if self.data.data_interface == D.sim1d_var_t:
                data_other = [{'length': l} for l in batch_lengths]
            else:
                data_other = [{} for _ in range(batch_size)]
        if divide_by_t:
            losses_base_tmp = torch.sum(torch.stack(losses_base_tmp)) / (n_t if n_t > 0 else 1)
        else:
            losses_base_tmp = torch.sum(torch.stack(losses_base_tmp))
        losses_tmp = torch.sum(torch.stack(losses_tmp))  # divide_by_t already handled in self.simulate(..)
        if include_data:
            return losses_tmp, losses_base_tmp, [data_gt, data_pred, data_other]
        else:
            return losses_tmp, losses_base_tmp
