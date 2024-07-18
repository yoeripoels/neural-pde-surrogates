from configs.parse_component import parse_base, load_config_modules, compose_config
from configs.train.defaults import base
from torch.nn import GELU, Tanh

base_args = {**base.default, **dict(
    base_resolution=(501, 96, 64),
    super_resolution=(501, 96, 64),
    experiment="twophase",
    time_window=25,
)}

base_args, parser = parse_base(base_args)  # re-parse with experiment name added

dataset_twophase = dict(
    object="PDE2DDataset",
    base_path=base_args.data_path,
    experiment=base_args.experiment,
    split_file="split",
    data_format="memmap",
    data_file="snapshots",
    conditioning="conditioning",
    spatial_conditioning='spatial_conditioning',
    name="twophase",
    preprocess=False,
    c_filter=[6],
)

trainer = dict(
    object="AutoregressivePushforwardTrainer",
    neighbors=3,
    time_window=base_args.time_window,
    base_resolution=base_args.base_resolution,
    super_resolution=base_args.super_resolution,
    device="cpu",
    batch_size=base_args.batch_size,
    nr_gt_steps=1,
    nw=0,
    num_epochs=10 * 50,
    lr_step_interval=25,
    unrolling=8,
    print_interval=4,
    test_interval=25,
    max_train_batches=float("inf"),
    max_test_batches=float("inf"),
    print_setting=dict(
        print_per_step=True,
    ),
    process_settings={}
)

model = dict(
    # wrapper args
    object="activation_wrapper",
    activation_final=Tanh(),
    enforce_spatial_cond=True,
    spatial_cond_channel=0,
    approx_volume_preserve=True,
    approx_volume_preserve_mode='individual_static',
    max_pct_dif=1 / 25,

    # regular model args
    model_class="EncProcDec",
    num_c=1,
    num_spatial_dims=2,
    time_window=base_args.time_window,
    data_structure="grid",
    processor_residual=False,

    encoder="enc_grid.ElementWise",

    activation=GELU(),

    processor="UNetModern",
    ch_mults=[2, 2, 1, 2],
    is_attn=[False for _ in range(4)],
    mid_attn=False,
    hidden_features=32,
    norm=True,
    use1x1=True,
    cond_mode="concat",
    padding_mode="circular",  # let's see

    decoder="dec_grid.TimeConvDense",
    dec_delta_mode='per_step',
    dec_kernel_size=5,
    dec_padding_mode="circular",
)

_, optimizer, lr_scheduler, _, criterion, _ = load_config_modules(base_args)

compose_config(parser,
               base_args=base_args,
               dataset=dataset_twophase,
               optimizer=optimizer.Adam,
               lr_scheduler=lr_scheduler.MultiStepLR,
               model=model,
               criterion=criterion.MSE_sum,
               trainer=trainer)