from abc import ABC, abstractmethod
import netCDF4
import numpy as np

class RAW_SST_Data(ABC):
    """Raw Sea Surface Temperature (SST) dataset"""

    def __init__(self):
        self.lon = None  # longitude, of shape (W,)
        self.lat = None  # latitude, of shape (H,)
        self.doy_sin = None  # day of year sine, of shape (T,)
        self.doy_cos = None  # day of year cosine, of shape (T,)
        self.land_mask = None  # mask accumulated over time (masks out the land pixels), of shape (H, W)
        self.sst = None  # sea surface temperature, of shape (T, H, W)
        self.mean = None  # mean of dataset
        self.std = None  # standard deviation of dataset

    @abstractmethod
    def load_data(self, data_path: str, n_samples: int, to_celsius: bool, standardize: bool) -> None:
        pass

def _get_lon_and_lat(lon_var_name: str, lat_var_name: str, data: netCDF4.Dataset):
    lon = data.variables[lon_var_name][:]
    lat = data.variables[lat_var_name][:]
    return lon, lat


def _get_sin_and_cos_doy(time_var_name: str, n_samples: int, data: netCDF4.Dataset):
    time_sec = data.variables[time_var_name]
    time_date = netCDF4.num2date(time_sec[:n_samples], time_sec.units)
    doy = np.array([d.timetuple().tm_yday for d in time_date])
    assert np.all((doy >= 1) & (doy <= 366)), "Day of the year values are out of range."
    doy_sin = np.sin(2 * np.pi * (doy / 365.25))
    doy_cos = np.cos(2 * np.pi * (doy / 365.25))
    return doy_sin, doy_cos


def _get_sst(
    sst_var_name: str, to_celsius: bool, standardize: bool, n_samples: int, data: netCDF4.Dataset
):
    sst = data.variables[sst_var_name][:n_samples]
    sst = sst - 273.15 if to_celsius else sst
    mean = None
    std = None
    if standardize:
        mean = np.mean(sst)
        std = np.std(sst)
        sst = (sst - mean) / std
        print(f"Dataset mean: {mean}")
        print(f"Dataset std: {std}")

    return sst, mean, std


def _get_land_mask(sst_var_name: str, n_samples: int, data: netCDF4.Dataset):
    H, W = data.variables[sst_var_name].shape[1:]
    land_mask = np.zeros((H, W))
    for idx in range(n_samples):
        # set non masked pixels to one
        temp = data.variables[sst_var_name][idx]
        land_mask[temp.mask == False] = 1

    return land_mask


class RAW_SST_Data_Mediterranean(RAW_SST_Data):
    """Raw SST data Mediterranean sea."""

    def __init__(
            self, data_path="./data/SST_L3_CMEMS_2006-2021_Mediterranean.nc", n_samples=None, standardize=True
    ):
        super().__init__()
        self.load_data(data_path, n_samples, standardize=standardize)
        self._change_spatial_res()

    def load_data(self, data_path, n_samples, to_celsius=True, standardize=True):
        # open the dataset
        data = netCDF4.Dataset(data_path, mode="r")

        # if `n_samples` is not defined take all samples
        if n_samples is None:
            n_samples = data.variables["sea_surface_temperature"].shape[0]

        # fetch the data
        self.lon, self.lat = _get_lon_and_lat("lon", "lat", data)
        self.doy_sin, self.doy_cos = _get_sin_and_cos_doy("time", n_samples, data)
        self.sst, self.mean, self.std = _get_sst("sea_surface_temperature", to_celsius,
                                                 standardize, n_samples, data)
        self.land_mask = _get_land_mask("sea_surface_temperature", n_samples, data)

    def _change_spatial_res(self):
        """Change the spatial resolution to (256, 256)"""

        # extrapolate latitude and remove longitude
        lat_delta = 0.0625
        lat_row = np.array(
            [self.lat[-1] + lat_delta * (x + 1) for x in range(3)]
        )  # extrapolate 3 values
        self.lat = np.hstack([self.lat, lat_row])
        self.lon = self.lon[:-17]  # remove last 17 values

        # change the spatial resolution of the land mask to (256, 256)
        self.land_mask = np.pad(
            self.land_mask, ((0, 3), (0, 0)), constant_values=0
        )  # pad 3 rows at the bottom
        self.land_mask = self.land_mask[:, :-17]  # remove the last 17 columns

        # change the spatial resolution of the sst map to (256, 256)
        ones_row = np.ones((self.sst.shape[0], 3, self.sst.shape[2]))
        masked_row = np.ma.masked_array(ones_row, mask=ones_row)
        self.sst = np.ma.concatenate(
            (self.sst, masked_row), axis=1
        )  # add 3 rows at the bottom
        self.sst = self.sst[:, :, :-17]  # remove the last 17 columns


class RAW_SST_Data_Adriatic(RAW_SST_Data):
    """Raw SST data Adriatic sea."""

    def __init__(self, data_path="./data/CHEMS_L3_SST_Adriatic.nc", n_samples=None, standardize=True):
        super().__init__()
        self.load_data(data_path, n_samples, standardize=standardize)
        self._change_spatial_res()

    def load_data(self, data_path, n_samples, to_celsius=True, standardize=True):
        # open the dataset
        data = netCDF4.Dataset(data_path, mode="r")

        # if `n_samples` is not defined take all samples
        if n_samples is None:
            n_samples = data.variables["adjusted_sea_surface_temperature"].shape[0]

        # fetch the data
        self.lon, self.lat = _get_lon_and_lat("lon", "lat", data)
        self.doy_sin, self.doy_cos = _get_sin_and_cos_doy("time", n_samples, data)
        self.sst, self.mean, self.std = _get_sst(
            "adjusted_sea_surface_temperature", to_celsius, standardize, n_samples, data
        )
        self.land_mask = _get_land_mask(
            "adjusted_sea_surface_temperature", n_samples, data
        )

    def _change_spatial_res(self):
        """Change the spatial resolution to (144, 180)"""
        # extrapolate latitude
        lat_delta = 0.05015564
        lat_row = np.array(
            [self.lat[-1] + lat_delta * (x + 1) for x in range(3)]
        )  # extrapolate 3 values
        self.lat = np.hstack([self.lat, lat_row])

        # change the spatial resolution of the land mask
        self.land_mask = np.pad(
            self.land_mask, ((0, 3), (0, 0)), constant_values=0
        )  # pad 3 rows at the bottom

        # change the spatial resolution of the sst map
        ones_row = np.ones((self.sst.shape[0], 3, self.sst.shape[2]))
        masked_row = np.ma.masked_array(ones_row, mask=ones_row)
        self.sst = np.ma.concatenate(
            (self.sst, masked_row), axis=1
        )  # add 3 rows at the bottom

        # FOR 144, 184
        lon_delta = 0.05004596
        lon_column = np.array(
            [self.lon[-1] + lon_delta * (x + 1) for x in range(4)]
        )  # extrapolate 4 values
        self.lon = np.hstack([self.lon, lon_column])  # append to the longitude array

        # Update the land mask by padding 3 rows at the bottom and 4 columns on the right
        self.land_mask = np.pad(
            self.land_mask, ((0, 0), (0, 4)), constant_values=0
        )  # pad rows and columns

        # Add 4 columns to the right with masked values for each time step
        ones_col = np.ones((self.sst.shape[0], self.sst.shape[1], 4))
        masked_col = np.ma.masked_array(ones_col, mask=ones_col)
        self.sst = np.ma.concatenate((self.sst, masked_col), axis=2)  # add 4 columns


class RAW_SST_Data_Atlantic(RAW_SST_Data):
    """Raw SST data Atlantic sea."""

    def __init__(self, data_path="./data/CMEMS_Atlantic_SST.nc", n_samples=None, standardize=True):
        super().__init__()
        self.load_data(data_path, n_samples, standardize=standardize)
        self._change_spatial_res()

    def load_data(self, data_path, n_samples, to_celsius=True, standardize=True):
        # open the dataset
        data = netCDF4.Dataset(data_path, mode="r")

        # if `n_samples` is not defined take all samples
        if n_samples is None:
            n_samples = data.variables["adjusted_sea_surface_temperature"].shape[0]

        # fetch the data
        self.lon, self.lat = _get_lon_and_lat("longitude", "latitude", data)
        self.doy_sin, self.doy_cos = _get_sin_and_cos_doy("time", n_samples, data)
        self.sst, self.mean, self.std = _get_sst(
            "adjusted_sea_surface_temperature", to_celsius, standardize, n_samples, data
        )
        self.land_mask = _get_land_mask(
            "adjusted_sea_surface_temperature", n_samples, data
        )

    def _change_spatial_res(self):
        """Change the spatial resolution to (256, 256)"""
        self.lon = self.lon[:256]
        self.lat = self.lat[:256]
        self.land_mask = self.land_mask[:256, :256]
        self.sst = self.sst[:, :256, :256]


def get_raw_sst_data(data_path: str, n_samples: int = None, standardize: bool = True) -> RAW_SST_Data:
    if "Mediterranean" in data_path:
        return RAW_SST_Data_Mediterranean(data_path, n_samples=n_samples, standardize=standardize)
    elif "Adriatic" in data_path:
        return RAW_SST_Data_Adriatic(data_path, n_samples=n_samples, standardize=standardize)
    elif "Atlantic" in data_path:
        return RAW_SST_Data_Atlantic(data_path, n_samples=n_samples, standardize=standardize)
    else:
        raise ValueError(f"Unkown dataset: {data_path}")