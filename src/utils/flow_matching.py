import torch


def expand_tensor_like(input_tensor: torch.Tensor, expand_to: torch.Tensor) -> torch.Tensor:
    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to).to(expand_to.device)


def cond_schedule(t: torch.Tensor):
    alpha_t = t
    sigma_t = 1 - t
    d_alpha_t = torch.ones_like(t)
    d_sigma_t = -torch.ones_like(t)
    return alpha_t, sigma_t, d_alpha_t, d_sigma_t


def path_sample(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
    alpha_t, sigma_t, d_alpha_t, d_sigma_t = cond_schedule(t)

    alpha_t = expand_tensor_like(input_tensor=alpha_t, expand_to=x_0)
    sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_0)
    d_alpha_t = expand_tensor_like(input_tensor=d_alpha_t, expand_to=x_0)
    d_sigma_t = expand_tensor_like(input_tensor=d_sigma_t, expand_to=x_0)

    x_t = sigma_t * x_0 + alpha_t * x_1
    dx_t = d_sigma_t * x_0 + d_alpha_t * x_1

    return x_t, dx_t


def path_sample_rectified(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
    alpha_t, sigma_t, _, _ = cond_schedule(t)

    # Expand t to match tensor shapes
    alpha_t = expand_tensor_like(alpha_t, x_0)
    sigma_t = expand_tensor_like(sigma_t, x_0)
    t_expanded = expand_tensor_like(t, x_0)
    
    x_t = sigma_t * x_0 + alpha_t * x_1
    dx_t = (x_1 - x_0) / t_expanded

    return x_t, dx_t