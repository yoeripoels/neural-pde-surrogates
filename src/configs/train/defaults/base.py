import os

default = dict(
    seed=42,
    time_window=25,
    batch_size=16,
    use_wandb=False,
    wandb_kwargs=dict(project="test-project", entity="neural-pde-surrogates"),
    num_c=1,
    data_path=os.environ["DATAROOT"] if "DATAROOT" in os.environ else "data",
    function_pre=None,
    function_post=None,
    experiment_path="experiments",
    experiment_name=None,
)