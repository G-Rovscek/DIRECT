import numpy as np
import torch.nn.functional as F
import torch
import random
from torch.utils.data import Dataset, DataLoader

from .raw_sst_data import get_raw_sst_data


SEED = 101
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def cloud_coverage(missing_mask, land_mask):
    return np.sum(1 - missing_mask) / np.sum(land_mask)


def _get_feat_to_idx(time_win, latlon_feat, mask_in_cond):
    """get feature to index mapping"""
    feat_to_idx = {}

    for dt in range(-(time_win // 2), time_win // 2 + 1):
        feat_to_idx[f"measurement_{dt}"] = len(feat_to_idx)

        if mask_in_cond:
            feat_to_idx[f"mask_{dt}"] = len(feat_to_idx)

    if latlon_feat:
        feat_to_idx["lon"] = len(feat_to_idx)
        feat_to_idx["lat"] = len(feat_to_idx)

    feat_to_idx["doy_cos"] = len(feat_to_idx)
    feat_to_idx["doy_sin"] = len(feat_to_idx)

    return feat_to_idx


def _construct_dataset(
        data_path,
        cloud_coverage_threshold,
        n_samples=None,
        standardize=True,
        time_win=3,
        mask_in_cond=False,
        max_delta=3,
):
    # Max delta not used in this construction
    _ = max_delta

    # Depricated | Not used
    latlon_feat = False
    
    # assert if time window is an even number
    assert time_win % 2 == 1, "time_win must be an odd number!"

    # get feature to index mappings
    feat_to_idx = _get_feat_to_idx(
        time_win, latlon_feat, mask_in_cond
    )
    idx_to_feat = {v: k for k, v in feat_to_idx.items()}

    # load the raw data
    raw_data = get_raw_sst_data(data_path, n_samples, standardize)
    lon, lat = raw_data.lon, raw_data.lat
    T, H, W = raw_data.sst.shape
    C = len(feat_to_idx)

    # iterate over the frames and construct the entire dataset
    n_skipped = 0
    data = {"observation": [], "missing_mask": []}
    for t in range(T):
        x = np.zeros((C, H, W))

        for dt in range(-(time_win // 2), time_win // 2 + 1):
            # clamp the time t between [0, T-1]
            t_clamped = np.clip(t + dt, 0, T - 1)

            # construct the current frame
            x[feat_to_idx[f"measurement_{dt}"], :, :] = raw_data.sst[t_clamped].filled(
                0
            )

            if mask_in_cond:
                x[feat_to_idx[f"mask_{dt}"], :, :] = 1 - raw_data.sst.mask[t_clamped]

        if latlon_feat:
            lon_scaled = 2 * (lon - lon.min()) / (lon.max() - lon.min()) - 1
            lat_scaled = 2 * (lat - lat.min()) / (lat.max() - lat.min()) - 1
            x[feat_to_idx["lon"], :, :] = lon_scaled.reshape(1, len(lon_scaled))
            x[feat_to_idx["lat"], :, :] = lat_scaled.reshape(len(lat_scaled), 1)

        x[feat_to_idx["doy_cos"], :, :] = raw_data.doy_cos[t]
        x[feat_to_idx["doy_sin"], :, :] = raw_data.doy_sin[t]

        # filter: skip samples with the fraction of missing values above the threshold
        missing_mask = 1 - raw_data.sst.mask[t] * raw_data.land_mask
        if cloud_coverage(missing_mask, raw_data.land_mask) >= cloud_coverage_threshold:
            n_skipped += 1
            continue

        data["observation"] += [torch.tensor(x, dtype=torch.float32)]
        data["missing_mask"] += [torch.tensor(missing_mask, dtype=torch.float32)]

    # convert to torch tensors
    land_mask = torch.Tensor(raw_data.land_mask)
    data["observation"] = torch.stack(data["observation"], dim=0)
    data["missing_mask"] = torch.stack(data["missing_mask"], dim=0)
    print(f"n_skipped: {n_skipped}", flush=True)
    print(f"n_samples: {data['observation'].shape[0]}", flush=True)
    return data, feat_to_idx, idx_to_feat, land_mask, raw_data.mean, raw_data.std


def _get_feat_to_idx_with_filled(time_win, mask_in_cond=False):
    feat_to_idx = {}

    for dt in range(-(time_win // 2), time_win // 2 + 1):
        feat_to_idx[f"measurement_{dt}"] = len(feat_to_idx)

    for dt in range(-(time_win // 2), time_win // 2 + 1):
        if mask_in_cond:
            feat_to_idx[f"mask_{dt}"] = len(feat_to_idx)
        if dt != 0:
            feat_to_idx[f"day_offset_{dt}"] = len(feat_to_idx)

    feat_to_idx["doy_sin"] = len(feat_to_idx)
    feat_to_idx["doy_cos"] = len(feat_to_idx)

    return feat_to_idx


def _get_init_feat_to_idx():
    feat_to_idx = {"sst": 0, "missing_mask": 1, "day_offset": 2}
    return feat_to_idx


def _get_inter_data(raw_data, feat_to_idx_init):
    T, H, W = raw_data.sst.shape
    C_raw = len(feat_to_idx_init)
    data = []
    for t in range(T):
        x = np.zeros((C_raw, H, W))
        x[feat_to_idx_init["sst"]] = raw_data.sst[t].filled(0)
        x[feat_to_idx_init["missing_mask"]] = 1 - raw_data.sst.mask[t] * raw_data.land_mask
        x[feat_to_idx_init["day_offset"], :, :] = 0.0

        data += [torch.tensor(x, dtype=torch.float32)]
    data = torch.stack(data)
    return data


def _fill_sst_future(data, feat_to_idx, max_delta):
    # Fill future data
    filled_future = data.clone()
    for t in reversed(range(data.shape[0] - 1)):
        curr = filled_future[t]
        next_ = filled_future[t + 1]

        missing = curr[feat_to_idx["missing_mask"]] == 0
        next_valid = next_[feat_to_idx["missing_mask"]] == 1

        fillable = missing & next_valid & (next_[feat_to_idx["day_offset"]] < max_delta)

        curr[feat_to_idx["day_offset"]] += 1
        curr[feat_to_idx["sst"]][fillable] = next_[feat_to_idx["sst"]][fillable]
        curr[feat_to_idx["day_offset"]][fillable] = next_[feat_to_idx["day_offset"]][fillable] + 1
        curr[feat_to_idx["day_offset"]] *= (curr[feat_to_idx["sst"]] != 0).float()

        curr[feat_to_idx["missing_mask"]][fillable] = 1.0
    return filled_future


def _fill_sst_past(data, feat_to_idx, max_delta):
    # Fill past data
    filled_past = data.clone()
    for t in range(1, data.shape[0]):
        curr = filled_past[t]
        prev = filled_past[t - 1]

        missing = curr[feat_to_idx["missing_mask"]] == 0
        prev_valid = prev[feat_to_idx["missing_mask"]] == 1

        fillable = missing & prev_valid & (prev[feat_to_idx["day_offset"]] > -max_delta)

        curr[feat_to_idx["day_offset"]] -= 1
        curr[feat_to_idx["sst"]][fillable] = prev[feat_to_idx["sst"]][fillable]
        curr[feat_to_idx["day_offset"]][fillable] = prev[feat_to_idx["day_offset"]][fillable] - 1
        curr[feat_to_idx["day_offset"]] *= (curr[feat_to_idx["sst"]] != 0).float()

        curr[feat_to_idx["missing_mask"]][fillable] = 1.0
    return filled_past


def _construct_dataset_with_filled(
        data_path,
        cloud_coverage_threshold,
        n_samples=None,
        standardize=True,
        time_win=3,
        mask_in_cond=False,
        max_delta=5,
):

    print(f"Max delta for filled dataset: {max_delta}")

    # assert if time window is an even number
    assert time_win % 2 == 1, "time_win must be an odd number!"

    raw_data = get_raw_sst_data(data_path, n_samples, standardize)
    feat_to_idx_init = _get_init_feat_to_idx()

    lon, lat = raw_data.lon, raw_data.lat
    T, H, W = raw_data.sst.shape

    # Create intermediate data
    data = _get_inter_data(raw_data, feat_to_idx_init)

    # Fill data with past and future
    filled_past = _fill_sst_past(data, feat_to_idx_init, max_delta)
    filled_future = _fill_sst_future(data, feat_to_idx_init, max_delta)

    feat_to_idx = _get_feat_to_idx_with_filled(
        time_win, mask_in_cond
    )
    idx_to_feat = {v: k for k, v in feat_to_idx.items()}
    C = len(feat_to_idx)

    # Construct final filled data
    filled_data = {"observation": [], "missing_mask": []}
    n_skipped = 0
    for t in range(T):
        x = np.zeros((C, H, W))

        # Filter by original missingness
        missing_mask = 1 - raw_data.sst.mask[t] * raw_data.land_mask
        if cloud_coverage(missing_mask, raw_data.land_mask) >= cloud_coverage_threshold:
            n_skipped += 1
            continue

        for dt in range(-(time_win // 2), time_win // 2 + 1):
            t_clamped = np.clip(t + dt, 0, T - 1)

            # Choose source: past or future
            if dt < 0:
                source = filled_past
            elif dt > 0:
                source = filled_future
            else:
                source = data  # use original data for center frame

            x[feat_to_idx[f"measurement_{dt}"]] = source[t_clamped, feat_to_idx_init["sst"]]
            if dt == 0:
                x[feat_to_idx["doy_cos"], :, :] = raw_data.doy_cos[t_clamped]
                x[feat_to_idx["doy_sin"], :, :] = raw_data.doy_sin[t_clamped]
            else:
                x[feat_to_idx[f"day_offset_{dt}"]] = source[t_clamped, feat_to_idx_init["day_offset"]]
            if mask_in_cond:
                x[feat_to_idx[f"mask_{dt}"], :, :] = (source[t_clamped, feat_to_idx_init["sst"]] != 0)

        filled_data["observation"] += [torch.tensor(x, dtype=torch.float32)]
        filled_data["missing_mask"] += [torch.tensor(missing_mask, dtype=torch.float32)]

    # Finalize tensors
    land_mask = torch.Tensor(raw_data.land_mask)
    filled_data["observation"] = torch.stack(filled_data["observation"], dim=0)
    filled_data["missing_mask"] = torch.stack(filled_data["missing_mask"], dim=0)

    print(f"n_skipped: {n_skipped}")
    print(f"n_samples: {filled_data['observation'].shape[0]}")
    return filled_data, feat_to_idx, idx_to_feat, land_mask, raw_data.mean, raw_data.std


def _get_feat_to_idx_onehot(time_win, max_delta, mask_in_cond=False):
    feat_to_idx = {}
    idx = 0
    for dt in range(-(time_win // 2), time_win // 2 + 1):
        feat_to_idx[f"measurement_{dt}"] = idx
        idx += 1
    for dt in range(-(time_win // 2), time_win // 2 + 1):
        if dt != 0:
            feat_to_idx[f"day_offset_{dt}_start"] = idx
            idx += max_delta

    if mask_in_cond:
        feat_to_idx['mask_0'] = idx
        idx += 1
    feat_to_idx["doy_sin"] = idx
    idx += 1
    feat_to_idx["doy_cos"] = idx

    return feat_to_idx


def _get_onehot_future(ff_tensor, max_delta):
    ff_tensor = torch.tensor(ff_tensor, dtype=torch.int64)
    valid_mask = ff_tensor != 0
    ff_shifted = torch.clamp(ff_tensor - 1, min=0)
    one_hot = F.one_hot(ff_shifted, num_classes=max_delta)
    one_hot[~valid_mask] = 0
    one_hot = one_hot.permute(2, 0, 1)
    return one_hot


def _get_onehot_past(fp_tensor, max_delta):
    fp_tensor = torch.tensor(fp_tensor, dtype=torch.int64)
    valid_mask = fp_tensor != 0  # 0 is "no data"
    fp_shifted = torch.clamp(fp_tensor + max_delta, 0, max_delta - 1)
    one_hot = F.one_hot(fp_shifted, num_classes=max_delta)  # shape [W, H, C]
    one_hot[~valid_mask] = 0
    one_hot = one_hot.permute(2, 0, 1)
    return one_hot


def _construct_dataset_with_onehot(
        data_path,
        cloud_coverage_threshold,
        n_samples=None,
        standardize=True,
        time_win=3,
        mask_in_cond=True,
        max_delta=3,
):

    print(f"Max delta for filled dataset: {max_delta}")

    # assert if time window is an even number
    assert time_win % 2 == 1, "time_win must be an odd number!"

    raw_data = get_raw_sst_data(data_path, n_samples, standardize)
    feat_to_idx_init = _get_init_feat_to_idx()

    T, H, W = raw_data.sst.shape

    # Create intermediate data
    data = _get_inter_data(raw_data, feat_to_idx_init)

    # Fill data with past and future
    filled_past = _fill_sst_past(data, feat_to_idx_init, max_delta)
    filled_future = _fill_sst_future(data, feat_to_idx_init, max_delta)

    feat_to_idx = _get_feat_to_idx_onehot(
        time_win=time_win, max_delta=max_delta, mask_in_cond=mask_in_cond
    )
    idx_to_feat = {v: k for k, v in feat_to_idx.items()}
    C = max(feat_to_idx.values()) + 1

    # Construct final filled data
    filled_data = {"observation": [], "missing_mask": []}
    n_skipped = 0

    for t in range(T):
        x = np.zeros((C, H, W))

        # Filter by original missingness
        missing_mask = 1 - raw_data.sst.mask[t] * raw_data.land_mask
        if cloud_coverage(missing_mask, raw_data.land_mask) >= cloud_coverage_threshold:
            n_skipped += 1
            continue

        for dt in range(-(time_win // 2), time_win // 2 + 1):
            t_clamped = np.clip(t + dt, 0, T - 1)

            # Choose source: past or future
            if dt < 0:
                source = filled_past
            elif dt > 0:
                source = filled_future
            else:
                source = data  # use original data for center frame

            x[feat_to_idx[f"measurement_{dt}"]] = source[t_clamped, feat_to_idx_init["sst"]]
            if dt == 0:
                x[feat_to_idx["doy_cos"], :, :] = raw_data.doy_cos[t_clamped]
                x[feat_to_idx["doy_sin"], :, :] = raw_data.doy_sin[t_clamped]
                if mask_in_cond:
                    x[feat_to_idx["mask_0"], :, :] = (source[t_clamped, feat_to_idx_init["sst"]] != 0)
            elif dt > 0:
                x[feat_to_idx[f"day_offset_{dt}_start"]:feat_to_idx[
                                                            f"day_offset_{dt}_start"] + max_delta] = _get_onehot_future(
                    source[t_clamped, feat_to_idx_init["day_offset"]], max_delta)
            elif dt < 0:
                x[feat_to_idx[f"day_offset_{dt}_start"]:feat_to_idx[
                                                            f"day_offset_{dt}_start"] + max_delta] = _get_onehot_past(
                    source[t_clamped, feat_to_idx_init["day_offset"]], max_delta)

        filled_data["observation"] += [torch.tensor(x, dtype=torch.float32)]
        filled_data["missing_mask"] += [torch.tensor(missing_mask, dtype=torch.float32)]

    # Finalize tensors
    land_mask = torch.Tensor(raw_data.land_mask)
    filled_data["observation"] = torch.stack(filled_data["observation"], dim=0)
    filled_data["missing_mask"] = torch.stack(filled_data["missing_mask"], dim=0)
    print(f"n_skipped: {n_skipped}", flush=True)
    print(f"n_samples: {filled_data['observation'].shape[0]}", flush=True)
    return filled_data, feat_to_idx, idx_to_feat, land_mask, raw_data.mean, raw_data.std


class SST_Dataset(Dataset):
    """Pytorch wrapper around the SST data."""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data["observation"].shape[0]

    def __getitem__(self, idx):
        observation, missing_mask = (
            self.data["observation"][idx],
            self.data["missing_mask"][idx],
        )
        if self.transform:
            observation = self.transform(observation)
        return observation, missing_mask, idx


def get_dataloaders(
        train_ratio=0.90,
        val_ratio=0.05,
        batch_size=8,
        shuffle=True,
        time_win=3,
        mask_in_cond=True,
        n_samples=None,
        standardize=True,
        cloud_coverage_threshold=1.0,
        data_path="./data/SST_L3_CMEMS_2006-2021_Adr.nc",
        max_delta=3,
        use_onehot=True,
):
    """
    Get dataloaders

    :param train_ratio: the ratio of samples used for training
    :param val_ratio: the ratio of samples used for validation
    :param batch_size: the number of samples in a single batch
    :param shuffle: whether to shuffle the **training set** or not
    :param time_win: number of time steps
    :param doy_feat: whether to utilize doy features or not
    :param latlon_feat: whether to utilize latitude, longitude features or not
    :param mask_in_cond: whether to utilize masks for days
    :param n_samples: the number of samples to load (if None load all)
    :param standardize: whether to standardize the data or not
    :param cloud_coverage_threshold: threshold used to filter observation fields with cloud coverage above it
    :param data_path: path to the dataset
    :param use_onehot: data with filled values and one-hot encoded masks
    :param max_delta: delta of fill window offset
    :return dataloaders: dictionary with keys: ["train", "val", "test"]
    :return metadata: metadata dictionary
    """

    if use_onehot:
        print("Using onehot dataset")
        dataset_function = _construct_dataset_with_onehot
    else:
        print("Using normal dataset")
        dataset_function = _construct_dataset

    # construct the dataset
    data, feat_to_idx, idx_to_feat, land_mask, data_mean, data_std = dataset_function(
        data_path=data_path,
        cloud_coverage_threshold=cloud_coverage_threshold,
        n_samples=n_samples,
        standardize=standardize,
        time_win=time_win,
        mask_in_cond=mask_in_cond,
        max_delta=max_delta,
    )

    # compute the number of training and validation samples
    n_samples = data["observation"].shape[0]
    n_train = round(train_ratio * n_samples)
    n_val = round(val_ratio * n_samples)

    # split into train, validation and test sets
    train_dataset = SST_Dataset(
        {
            "observation": data["observation"][:n_train, :, :, :],
            "missing_mask": data["missing_mask"][:n_train, :, :],
        }
    )

    val_dataset = SST_Dataset(
        {
            "observation": data["observation"][n_train: n_train + n_val, :, :, :],
            "missing_mask": data["missing_mask"][n_train: n_train + n_val, :, :],
        }
    )

    test_dataset = SST_Dataset(
        {
            "observation": data["observation"][n_train + n_val:, :, :, :],
            "missing_mask": data["missing_mask"][n_train + n_val:, :, :],
        }
    )

    # construct data loaders
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        "val": DataLoader(val_dataset, batch_size=batch_size),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }

    # gather metadata
    metadata = {
        "land_mask": land_mask,
        "feat_to_idx": feat_to_idx,
        "idx_to_feat": idx_to_feat,
        "time_win": time_win,
        "data_mean": data_mean,
        "data_std": data_std,
        "max_delta": max_delta,
    }

    return dataloaders, metadata


class MaskSampler:
    """Class used to sample (cloud) masks from the dataset"""

    def __init__(self, dataset: SST_Dataset) -> None:
        # init. device to CPU
        self._device = torch.device("cpu")

        # store the dataset and pre-compute indices
        self.dataset = dataset
        self.indices = len(self.dataset)

    def __call__(self, batch_size=1):
        return self.sample_mask(batch_size=batch_size)

    def to(self, device: torch.device):
        self._device = device

    def sample_mask(self, batch_size=1):
        """Sample a cloud mask"""
        idx = torch.randint(0, self.indices, (batch_size,))
        _, mask, *_ = self.dataset[idx]
        # Shape: [B x W x H]
        return mask.to(self._device), idx

    def sample_valid_mask(self, land_mask, missing_mask, max_attempts=100):
        """
        Sample a valid mask, i.e., mask which conceals at least one pixel,
        mask which keeps at least one visible pixel
        """
        B, W, H = missing_mask.shape
        masks = []
        indices = []

        land_mask = land_mask.unsqueeze(0)

        for i in range(B):
            # Attempt sampling for each sample in the batch
            for _ in range(max_attempts):
                mask, idx = self.sample_mask(batch_size=1)  # [1, W, H]
                mask = mask.squeeze(0).to(missing_mask.device)

                mask_vis = missing_mask[i] * land_mask * mask
                mask_hid = missing_mask[i] * land_mask * (1 - mask)

                if torch.sum(mask_vis) > 0 and torch.sum(mask_hid) > 0:
                    masks.append(mask)
                    indices.append(idx.item())
                    break
            else:
                raise RuntimeError(f"Failed to sample valid mask for sample {i} in {max_attempts} attempts")

        masks = torch.stack(masks, dim=0)  # [B, W, H]
        indices = torch.tensor(indices, device=self._device)
        return masks, indices

    def sample_valid_mask_batch(self, land_mask, missing_mask, pool_size=256, max_attempts=10):
        B, W, H = missing_mask.shape
        device = missing_mask.device

        for attempt in range(max_attempts):
            # Sample candidate masks
            masks, idxs = self.sample_mask(batch_size=pool_size)  # [pool_size, W, H], [pool_size]
            masks = masks.to(device)
            idxs = idxs.to(device)

            # Expand candidate masks for all batch samples
            mask_pool = masks.unsqueeze(0).expand(B, -1, -1, -1)  # [B, pool_size, W, H]
            land_mask_exp = land_mask.view(1, 1, W, H)
            miss_exp = missing_mask.unsqueeze(1)  # [B, 1, W, H]

            # Compute visibility masks
            vis = miss_exp * land_mask_exp * mask_pool
            hid = miss_exp * land_mask_exp * (1 - mask_pool)

            # Check for validity per candidate: (visible and hidden pixels exist)
            valid = (vis.sum(dim=[2, 3]) > 0) & (hid.sum(dim=[2, 3]) > 0)  # [B, pool_size]

            # Sample a valid index *randomly* per batch element
            valid_idxs = [torch.nonzero(valid[i], as_tuple=False).squeeze(1) for i in range(B)]

            # Ensure all have at least one valid index
            if all(len(vi) > 0 for vi in valid_idxs):
                sampled_indices = torch.tensor([
                    vi[torch.randint(len(vi), (1,))] for vi in valid_idxs
                ], device=device)

                selected_masks = masks[sampled_indices]
                selected_ids = idxs[sampled_indices]

                return selected_masks, selected_ids

        raise RuntimeError(f"Failed to sample valid masks in {max_attempts} attempts")