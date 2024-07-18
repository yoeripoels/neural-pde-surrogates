from utils.attr import getattr_nested
from pdes import PDE
from torch import nn
import torch
import models
import einops


def activation_wrapper(model_class: str, activation_final: nn.Module,
                       enforce_spatial_cond=False,
                       spatial_cond_channel=0,
                       approx_volume_preserve=False,
                       approx_volume_preserve_mode='block',
                       max_pct_dif=1,
                       *args, **kwargs):
    modules_to_check = [models.enc_proc_dec_components, models, models.common]
    modeltype = None
    for module in modules_to_check:
        if (model_init := getattr_nested(module, model_class)) is not False:
            modeltype = model_init
            break
    if modeltype is False:
        raise ValueError(f"Model {model_class} not found")

    def __apply_spatial_cond(spatial_cond, u):
        to_zero = spatial_cond[:, spatial_cond_channel]
        # repeat our to_zero to the input shape
        to_zero = einops.repeat(to_zero, 'b x y -> b c tw x y', c=u.shape[1], tw=u.shape[2])

        # now zero out the desired values
        return u - (to_zero * u)

    def new_forward(*a, **b):
        u = activation_final(modeltype.forward(*a, **b))
        if enforce_spatial_cond:
            u = __apply_spatial_cond(b["spatial_cond"], u)
        if approx_volume_preserve:
            u_prev = b["x"] if "x" in b else a[1]
            num_spatial_dims = len(u_prev.shape) - 3
            if approx_volume_preserve_mode == 'block':
                prev_totals = torch.sum(u_prev[:, :, -1, ...], axis=list(range(2, 2 + num_spatial_dims)))
                new_meantotals = torch.mean(torch.sum(u, axis=list(range(3, 3 + num_spatial_dims))), axis=2)
                dif_coefs = (1 - new_meantotals / prev_totals) * 100
                dif_coefs = torch.tanh(dif_coefs / max_pct_dif) / 100 * max_pct_dif
                rescalings = 1 - dif_coefs
                # do some broadcasting by hand
                if num_spatial_dims == 2:
                    new_meantotals, prev_totals, rescalings = \
                        [einops.repeat(tensor, 'b c -> b c tw x y', tw=u.shape[2], x=u.shape[3], y=u.shape[4]) for tensor in
                         [new_meantotals, prev_totals, rescalings]]
                else:
                    raise ValueError(f"{num_spatial_dims} spatial dims not supported for approx volume preserve")
                u = (u / new_meantotals) * (prev_totals * rescalings)
            elif approx_volume_preserve_mode == 'individual':
                new_meantotals = torch.sum(u, axis=list(range(3, 3 + num_spatial_dims)))

                # iteratively apply
                rescalings_all = torch.zeros_like(new_meantotals)
                prev_totals_all = torch.zeros_like(new_meantotals)
                prev_totals = torch.sum(u_prev[:, :, -1, ...], axis=list(range(2, 2 + num_spatial_dims)))
                prev_totals_all[:, :, 0] = prev_totals
                for i in range(new_meantotals.shape[-1]):
                    step_meantotals = new_meantotals[:, :, i]
                    dif_coefs = (1 - step_meantotals / prev_totals) * 100
                    dif_coefs = torch.tanh(dif_coefs / max_pct_dif) / 100 * max_pct_dif
                    rescalings = 1 - dif_coefs
                    rescalings_all[:, :, i] = rescalings
                    if i < new_meantotals.shape[-1] - 1:
                        prev_totals_all[:, :, i + 1] = (rescalings * prev_totals)
                        prev_totals = rescalings * prev_totals

                # do some broadcasting by hand
                if num_spatial_dims == 2:
                    new_meantotals, prev_totals_all, rescalings_all = \
                        [einops.repeat(tensor, 'b c tw -> b c tw x y', x=u.shape[3], y=u.shape[4]) for tensor in
                         [new_meantotals, prev_totals_all, rescalings_all]]
                else:
                    raise ValueError(f"{num_spatial_dims} spatial dims not supported for approx volume preserve")
                u = (u / new_meantotals) * (rescalings_all * prev_totals_all)
            elif approx_volume_preserve_mode == 'individual_static':
                new_meantotals = torch.sum(u, axis=list(range(3, 3 + num_spatial_dims)))

                # vectorized application
                prev_totals_all = torch.sum(u_prev[:, :, -1, ...], axis=list(range(2, 2 + num_spatial_dims)))

                prev_totals_all = einops.repeat(prev_totals_all, 'b c -> b c tw', tw=u.shape[2])
                max_pct_dif_all = torch.ones_like(new_meantotals) * max_pct_dif
                max_pct_dif_all = torch.cumsum(max_pct_dif_all, dim=2)

                dif_coefs = (1 - new_meantotals / prev_totals_all) * 100
                dif_coefs = torch.tanh(dif_coefs / max_pct_dif_all) / 100 * max_pct_dif_all
                rescalings_all = 1 - dif_coefs

                # do some broadcasting by hand
                if num_spatial_dims == 2:
                    new_meantotals, prev_totals_all, rescalings_all = \
                        [einops.repeat(tensor, 'b c tw -> b c tw x y', x=u.shape[3], y=u.shape[4]) for tensor in
                         [new_meantotals, prev_totals_all, rescalings_all]]
                else:
                    raise ValueError(f"{num_spatial_dims} spatial dims not supported for approx volume preserve")
                u = (u / new_meantotals) * (rescalings_all * prev_totals_all)
            else:
                raise ValueError(f"Unrecognized approx_volume_preserve_mode '{approx_volume_preserve_mode}'")
            if enforce_spatial_cond:  # re-apply spatial cond after this renormalization
                u = __apply_spatial_cond(b["spatial_cond"], u)
        return u

    return type(f'ActWrapper-{model_class}', (modeltype,), {'forward': new_forward})(*args, **kwargs)
