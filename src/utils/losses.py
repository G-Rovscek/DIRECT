import torch

def masked_mean_squared_error(target, rec, mask):
    """Compute mean squared error (MSE) wrt to the binary mask"""
    se = masked_squared_error(target, rec, mask)
    # print(f"Sum of SE: {torch.sum(se)}" )
    # print(f"Non zero in SE: {torch.count_nonzero(se)}")
    # print(f"Non zero in mask: {torch.count_nonzero(mask)}")
    return torch.sum(se) / (torch.count_nonzero(mask))


def masked_squared_error(target, rec, mask):
    """Compute squared error (SE) wrt to the binary mask"""
    return ((target - rec) * mask).square()


def compute_mse(target, rec, missing_mask, land_mask, sampled_mask):
    """
    Compute mean squared error over the following three regions:
    all pixels, visible pixels, masked pixels
    """
    loss_dict = {}

    # compute mse over all pixels
    M_all = missing_mask * land_mask
    loss_dict["mse_all"] = masked_mean_squared_error(target, rec, M_all)
    loss_dict["M_all"] = M_all

    # compute mse over visible pixels
    M_vis = M_all * sampled_mask
    loss_dict["mse_vis"] = masked_mean_squared_error(target, rec, M_vis)
    loss_dict["M_vis"] = M_vis

    # compute mse over hidden pixels
    M_hid = M_all * (1 - sampled_mask)
    loss_dict["mse_hid"] = masked_mean_squared_error(target, rec, M_hid)
    loss_dict["M_hid"] = M_hid

    # add the loss key
    loss_dict["loss_all"] = loss_dict["mse_all"]
    return loss_dict
