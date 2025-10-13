from netCDF4 import Dataset
import numpy as np
import pandas as pd
import torch
from haversine import haversine_vector, Unit


class LoadData:


    def __init__(self, train_ratio, valid_ratio, input_seq_len, output_seq_len):
        self.X, self.Y, self.num, self.num_nodes, self.features, self.coordinates, self.mete_long, self.mete_lat = self._load()
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self._split(int(self.train_ratio * self.num), int((self.valid_ratio + self.train_ratio) * self.num))
        self.adj = self._getAdj()

    # -----------------------------
    # Utils
    # -----------------------------
    def _fill_missing(self, arr):
        """Linear interpolation (both directions) then ffill/bfill. No final zero fill."""
        arr = np.array(arr, dtype=float, copy=True)
        T = arr.shape[0]
        if arr.ndim == 1:
            s = pd.Series(arr)
            s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
            return s.values
        flat = arr.reshape(T, -1)
        df = (
            pd.DataFrame(flat)
            .interpolate(method="linear", axis=0, limit_direction="both")
            .ffill(axis=0)
            .bfill(axis=0)
        )
        return df.values.reshape(arr.shape)

    def get_azimuth(self):
       
        azimuth = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                dlat = self.coordinates[i, 0] - self.coordinates[j, 0]
                dlon = self.coordinates[i, 1] - self.coordinates[j, 1]
                if dlat > 0 and dlon > 0:
                    azimuth[i, j] = 3
                elif dlat > 0 and dlon < 0:
                    azimuth[i, j] = 1
                elif dlat < 0 and dlon < 0:
                    azimuth[i, j] = 4
                elif dlat < 0 and dlon > 0:
                    azimuth[i, j] = 2
                elif dlat > 0 and dlon == 0:
                    azimuth[i, j] = 5
                elif dlat < 0 and dlon == 0:
                    azimuth[i, j] = 6
                elif dlat == 0 and dlon < 0:
                    azimuth[i, j] = 7
                elif dlat == 0 and dlon > 0:
                    azimuth[i, j] = 8
                if int(azimuth[i, j]) % 2 == 1:
                    azimuth[j, i] = azimuth[i, j] + 1
                else:
                    azimuth[j, i] = azimuth[i, j] - 1
        return azimuth

    # -----------------------------
    # Core IO
    # -----------------------------
    def _load(self):
     
        time_key_candidates = ("time", "valid_time", "datetime")
        longitude_2023 = latitude_2023 = None
        time_2023 = t2m_2023 = u10_2023 = v10_2023 = d2m_2023 = sp_2023 = tp_2023 = None

        for month in range(1, 12 + 1):
            file_path = f"/home/23meteroloy_data/2023{str(month).zfill(2)}.nc"
            nc_2023 = Dataset(file_path)
            time_key = None
            for key in time_key_candidates:
                if key in nc_2023.variables:
                    time_key = key
                    break
            if time_key is None:
                raise KeyError(f"No time variable found in 2023 file for month {month}")

            if month == 1:
                longitude_2023 = np.array(nc_2023.variables["longitude"][:])
                latitude_2023 = np.array(nc_2023.variables["latitude"][:])
                time_2023 = np.array(nc_2023.variables[time_key][:])
                t2m_2023 = np.array(nc_2023.variables["t2m"][:])
                u10_2023 = np.array(nc_2023.variables["u10"][:])
                v10_2023 = np.array(nc_2023.variables["v10"][:])
                d2m_2023 = np.array(nc_2023.variables["d2m"][:])
                sp_2023 = np.array(nc_2023.variables["sp"][:])
                tp_2023 = np.array(nc_2023.variables["tp"][:])
            else:
                time_2023 = np.concatenate((time_2023, np.array(nc_2023.variables[time_key][:])), axis=0)
                t2m_2023 = np.concatenate((t2m_2023, np.array(nc_2023.variables["t2m"][:])), axis=0)
                u10_2023 = np.concatenate((u10_2023, np.array(nc_2023.variables["u10"][:])), axis=0)
                v10_2023 = np.concatenate((v10_2023, np.array(nc_2023.variables["v10"][:])), axis=0)
                d2m_2023 = np.concatenate((d2m_2023, np.array(nc_2023.variables["d2m"][:])), axis=0)
                sp_2023 = np.concatenate((sp_2023, np.array(nc_2023.variables["sp"][:])), axis=0)
                tp_2023 = np.concatenate((tp_2023, np.array(nc_2023.variables["tp"][:])), axis=0)

        # ---- 2024 meteorology (month by month), concat after 2023 ----
        longitude = latitude = None
        time = t2m = u10 = v10 = d2m = sp = tp = None

        for month in range(1, 12 + 1):
            file_path = f"/home/24meteroloy_data/2024{str(month).zfill(2)}.nc"
            nc_2024 = Dataset(file_path)
            time_key = None
            for key in time_key_candidates:
                if key in nc_2024.variables:
                    time_key = key
                    break
            if time_key is None:
                raise KeyError(f"No time variable found in 2024 file for month {month}")

            if month == 1:
                longitude = np.array(nc_2024.variables["longitude"][:])
                latitude = np.array(nc_2024.variables["latitude"][:])

                time = np.concatenate((time_2023, np.array(nc_2024.variables[time_key][:])), axis=0)
                t2m = np.concatenate((t2m_2023, np.array(nc_2024.variables["t2m"][:])), axis=0)
                u10 = np.concatenate((u10_2023, np.array(nc_2024.variables["u10"][:])), axis=0)
                v10 = np.concatenate((v10_2023, np.array(nc_2024.variables["v10"][:])), axis=0)
                d2m = np.concatenate((d2m_2023, np.array(nc_2024.variables["d2m"][:])), axis=0)
                sp = np.concatenate((sp_2023, np.array(nc_2024.variables["sp"][:])), axis=0)
                tp = np.concatenate((tp_2023, np.array(nc_2024.variables["tp"][:])), axis=0)
            else:
                time = np.concatenate((time, np.array(nc_2024.variables[time_key][:])), axis=0)
                t2m = np.concatenate((t2m, np.array(nc_2024.variables["t2m"][:])), axis=0)
                u10 = np.concatenate((u10, np.array(nc_2024.variables["u10"][:])), axis=0)
                v10 = np.concatenate((v10, np.array(nc_2024.variables["v10"][:])), axis=0)
                d2m = np.concatenate((d2m, np.array(nc_2024.variables["d2m"][:])), axis=0)
                sp = np.concatenate((sp, np.array(nc_2024.variables["sp"][:])), axis=0)
                tp = np.concatenate((tp, np.array(nc_2024.variables["tp"][:])), axis=0)

        # ---- Pollutants (2023 & 2024 CSVs) ----
        pol_2023_dir = "/home/2023pollute_data"
        pol_2024_dir = "/home/2024pollute_data"
        stations = [ ]

        def _read_and_interp(dirpath, name):
            year = "2024" if "2024" in dirpath else "2023"
            df = pd.read_csv(f"{dirpath}/{year}_{name}.csv")
            df = df[stations].interpolate(method="linear", limit_direction="forward")
            df = df.interpolate(method="linear", limit_direction="backward")
            return df

        pol_names = ["AQI", "CO", "NO2", "O3", "PM10", "PM25", "SO2"]
        pol_2023 = {n: _read_and_interp(pol_2023_dir, n) for n in pol_names}
        pol_2024 = {n: _read_and_interp(pol_2024_dir, n) for n in pol_names}
        pol_all = {n: np.concatenate([pol_2023[n].to_numpy(), pol_2024[n].to_numpy()], axis=0) for n in pol_names}

        # ---- Station coordinates ----
        dstation = pd.read_excel("/home/station.xlsx")
        coordinates = np.array(dstation[["latitude", "longitude"]]).reshape(-1, 2)


        T_total = t2m.shape[0]
        N_st = coordinates.shape[0]

        dt2m = np.zeros((T_total, N_st), dtype=float)
        du10 = np.zeros_like(dt2m)
        dv10 = np.zeros_like(dt2m)
        dd2m = np.zeros_like(dt2m)
        dsp  = np.zeros_like(dt2m)
        dtp  = np.zeros_like(dt2m)

        wlongitude = [int(np.argmin(np.abs(longitude - coordinates[i, 1]))) for i in range(N_st)]
        wlatitude  = [int(np.argmin(np.abs(latitude  - coordinates[i, 0]))) for i in range(N_st)]

        for k in range(T_total):
            for j in range(N_st):
                lat_idx = wlatitude[j]
                lon_idx = wlongitude[j]
                dt2m[k, j] = t2m[k, lat_idx, lon_idx]
                du10[k, j] = u10[k, lat_idx, lon_idx]
                dv10[k, j] = v10[k, lat_idx, lon_idx]
                dd2m[k, j] = d2m[k, lat_idx, lon_idx]
                dsp[k, j]  = sp[k,  lat_idx, lon_idx]
                dtp[k, j]  = tp[k,  lat_idx, lon_idx]

        # Interpolate meteorology only (no zero fallback)
        dt2m = self._fill_missing(dt2m)
        du10 = self._fill_missing(du10)
        dv10 = self._fill_missing(dv10)
        dd2m = self._fill_missing(dd2m)
        dsp  = self._fill_missing(dsp)
        dtp  = self._fill_missing(dtp)

        # ---- Pollutant matrices ----
        daqi  = pol_all["AQI"]
        dco   = pol_all["CO"]
        dno2  = pol_all["NO2"]
        do3   = pol_all["O3"]      # no np.nan_to_num; keep standard behavior
        dpm10 = pol_all["PM10"]
        dpm25 = pol_all["PM25"]
        dso2  = pol_all["SO2"]

        # ---- Feature stacking: F=13 ----
        # order: t2m, u10, v10, d2m, sp, tp, AQI, CO, NO2, O3, PM10, SO2, PM25
        feature_list = [dt2m, du10, dv10, dd2m, dsp, dtp, daqi, dco, dno2, do3, dpm10, dso2, dpm25]
        num_features = len(feature_list)
        T_pol = daqi.shape[0]
        assert T_pol == T_total, "Meteorology and pollutant time lengths differ"

        data_final = np.zeros((N_st, T_total, num_features), dtype=float)
        for i in range(N_st):
            for f_idx, feat in enumerate(feature_list):
                data_final[i, :, f_idx] = feat[:, i]

        # ---- Target Y: PM2.5 (last column) ----
        Y = data_final[:, :, -1].reshape(-1, 1)

        # ---- Per-station min–max normalization for X only ----
        data_normal = []
        for i in range(N_st):
            data_i_norm, _, _ = self._Maxmin(data_final[i, :, :])
            data_normal.append(data_i_norm)

        data = np.concatenate(data_normal, axis=0)      # (N*T, F)
        Y = Y.reshape(N_st, -1).T                       # (T, N)
        data = data.reshape(N_st, -1, num_features)     # (N, T, F)
        X = data.transpose(1, 0, 2)                     # (T, N, F)

        return X, Y, X.shape[0], X.shape[1], X.shape[2], coordinates, longitude, latitude

    # -----------------------------
    # Graph
    # -----------------------------
    def _getAdj(self, k=4, sigma=None):
        """
        Geographic KNN adjacency with Gaussian weights and symmetric normalization.
        Returns FloatTensor (N, N); moved to GPU if available.
        """
        coords = [(float(lat), float(lon)) for lat, lon in self.coordinates]
        dist_mat = haversine_vector(coords, coords, Unit.KILOMETERS, comb=True)
        N = dist_mat.shape[0]

        if sigma is None:
            sorted_d = np.sort(dist_mat + np.eye(N) * 1e6, axis=1)
            knn_dists = sorted_d[:, 1:k + 1]
            sigma = float(np.mean(knn_dists))

        A = np.zeros_like(dist_mat, dtype=float)
        for i in range(N):
            knn_idx = np.argsort(dist_mat[i])[1:k + 1]
            A[i, knn_idx] = 1
        A = np.maximum(A, A.T)

        A_weighted = np.exp(-(dist_mat ** 2) / (2 * sigma ** 2)) * (A > 0)

        deg = np.sum(A_weighted, axis=1)
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A_weighted @ D_inv_sqrt

        A_tensor = torch.from_numpy(A_norm.astype(np.float32))
        if torch.cuda.is_available():
            A_tensor = A_tensor.cuda()
        return A_tensor

    # -----------------------------
    # Normalization helpers
    # -----------------------------
    def _Maxmin(self, data):
        """Per-feature min–max normalization; return (normed, max, min)."""
        maxdata = np.max(data, axis=0)
        mindata = np.min(data, axis=0)
        denominator = maxdata - mindata
        for i in range(denominator.shape[0]):
            if denominator[i] == 0:
                denominator[i] += 1e-8
        return (data - mindata) / denominator, maxdata, mindata

    def test_Maxmin(self, data, maxdata, mindata):
        """Apply given min–max to new data."""
        return (data - mindata) / (maxdata - mindata)

    def deMaxmin(self, data, maxdata, mindata):
        """Invert min–max normalization."""
        return (data * (maxdata - mindata)) + mindata

    # -----------------------------
    # Split & batching
    # -----------------------------
    def _split(self, train, valid):
        """Create index ranges and materialize train/valid/test sliding windows."""
        train_set = range(self.input_seq_len + self.output_seq_len - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.num)

        self.train = self._batchify(train_set, self.output_seq_len)
        self.valid = self._batchify(valid_set, self.output_seq_len)
        self.test  = self._batchify(test_set,  self.output_seq_len)

    def _batchify(self, idx_set, horizon):
        """Materialize X/Y windows for provided index set."""
        n = len(idx_set)
        dataX = torch.empty((n, self.input_seq_len, self.num_nodes, self.features), dtype=torch.float32, pin_memory=True)
        dataY = torch.empty((n, self.output_seq_len, self.num_nodes), dtype=torch.float32, pin_memory=True)

        for i in range(n):
            end = idx_set[i] - self.output_seq_len + 1
            start = end - self.input_seq_len
            dataX[i, :, :, :] = torch.from_numpy(self.X[start:end, :, :])
            if self.output_seq_len == 1:
                dataY[i, 0, :] = torch.from_numpy(self.Y[end, :])
            else:
                dataY[i, :, :] = torch.from_numpy(self.Y[end:end + self.output_seq_len, :])
        return [dataX, dataY]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        """Yield mini-batches of (X, Y)."""
        length = len(inputs)
        index = torch.randperm(length) if shuffle else torch.arange(length)
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            yield X, Y
            start_idx += batch_size


if __name__ == "__main__":
    train_ratio, valid_ratio, input_seq_len, output_seq_len = 0.6, 0.2, 24, 24
    Data = LoadData(train_ratio, valid_ratio, input_seq_len, output_seq_len)
    print("X shape:", Data.X.shape, "Y shape:", Data.Y.shape)
