def count_parameters(model, provided_as_params=False):
    if provided_as_params:
        return sum(p.numel() for p in model if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
