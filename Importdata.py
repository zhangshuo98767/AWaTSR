
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import torch
from haversine import haversine_vector, Unit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class LoadData():
    def __init__(self, train_ratio, valid_ratio, input_seq_len, output_seq_len):
        # 这些数据都是标准化过的
        self.X, self.Y, self.num, self.num_nodes, self.features, self.coordinates, self.mete_long, self.mete_lat = self._load()
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self._split(int(self.train_ratio * self.num), int((self.valid_ratio + self.train_ratio) * self.num))
        self.adj = self._getAdj()

    def _fill_missing(self, arr):
        arr = np.array(arr, dtype=float, copy=True)
        T = arr.shape[0]
        if arr.ndim == 1:
            s = pd.Series(arr)
            s = s.interpolate(method='linear', limit_direction='both').ffill().bfill()
            return s.fillna(0).values
        flat = arr.reshape(T, -1)
        df = pd.DataFrame(flat).interpolate(method='linear', axis=0, limit_direction='both').ffill(axis=0).bfill(axis=0).fillna(0)
        return df.values.reshape(arr.shape)

    def get_azimuth(self):
    
        azimuth = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.coordinates[i, 0] - self.coordinates[j, 0] > 0 and self.coordinates[i, 1] - self.coordinates[j, 1] > 0:
                    azimuth[i, j] = 3
                elif self.coordinates[i, 0] - self.coordinates[j, 0] > 0 and self.coordinates[i, 1] - self.coordinates[j, 1] < 0:
                    azimuth[i, j] = 1
                elif self.coordinates[i, 0] - self.coordinates[j, 0] < 0 and self.coordinates[i, 1] - self.coordinates[j, 1] < 0:
                    azimuth[i, j] = 4
                elif self.coordinates[i, 0] - self.coordinates[j, 0] < 0 and self.coordinates[i, 1] - self.coordinates[j, 1] > 0:
                    azimuth[i, j] = 2
                elif self.coordinates[i, 0] - self.coordinates[j, 0] > 0 and self.coordinates[i, 1] - self.coordinates[j, 1] == 0:
                    azimuth[i, j] = 5
                elif self.coordinates[i, 0] - self.coordinates[j, 0] < 0 and self.coordinates[i, 1] - self.coordinates[j, 1] == 0:
                    azimuth[i, j] = 6
                elif self.coordinates[i, 0] - self.coordinates[j, 0] == 0 and self.coordinates[i, 1] - self.coordinates[j, 1] < 0:
                    azimuth[i, j] = 7
                elif self.coordinates[i, 0] - self.coordinates[j, 0] == 0 and self.coordinates[i, 1] - self.coordinates[j, 1] > 0:
                    azimuth[i, j] = 8
                if azimuth[i, j] % 2 == 1:
                    azimuth[j, i] = azimuth[i, j] + 1
                else:
                    azimuth[j, i] = azimuth[i, j] - 1
        return torch.tensor(azimuth)

    # 无量纲化
    def dimensionlessProcessing(self, df_values):
        scaler = StandardScaler()
        res = scaler.fit_transform(df_values)
        return pd.DataFrame(res)

    def Calculate_cos_simi(self):
        # 注意：需要先给 self.POI 赋值
        cosine_similarities = cosine_similarity(self.POI)
        print('cosine_similarities:', cosine_similarities)
        return cosine_similarities

    def _load(self):
        # ---- 读取 2023 气象（逐月拼接）----
        for i in range(1, 13):
            file_path = f'/home/23meteroloy_data/2023{str(i).zfill(2)}.nc'
            nc_obj_2022 = Dataset(file_path)
            time_key = None
            for key in ['time', 'valid_time', 'datetime']:
                if key in nc_obj_2022.variables:
                    time_key = key
                    break
            if time_key is None:
                raise KeyError("No time variable found in 23meteroloy_data file for month " + str(i))
            if i == 1:
                longitude_2022 = np.array(nc_obj_2022.variables['longitude'][:])
                latitude_2022 = np.array(nc_obj_2022.variables['latitude'][:])
                time_2022 = np.array(nc_obj_2022.variables[time_key][:])
                t2m_2022 = np.array(nc_obj_2022.variables['t2m'][:])
                u10_2022 = np.array(nc_obj_2022.variables['u10'][:])
                v10_2022 = np.array(nc_obj_2022.variables['v10'][:])
                d2m_2022 = np.array(nc_obj_2022.variables['d2m'][:])
                sp_2022 = np.array(nc_obj_2022.variables['sp'][:])
                tp_2022 = np.array(nc_obj_2022.variables['tp'][:])
            else:
                time_2022 = np.concatenate((time_2022, np.array(nc_obj_2022.variables[time_key][:])), axis=0)
                t2m_2022 = np.concatenate((t2m_2022, np.array(nc_obj_2022.variables['t2m'][:])), axis=0)
                u10_2022 = np.concatenate((u10_2022, np.array(nc_obj_2022.variables['u10'][:])), axis=0)
                v10_2022 = np.concatenate((v10_2022, np.array(nc_obj_2022.variables['v10'][:])), axis=0)
                d2m_2022 = np.concatenate((d2m_2022, np.array(nc_obj_2022.variables['d2m'][:])), axis=0)
                sp_2022 = np.concatenate((sp_2022, np.array(nc_obj_2022.variables['sp'][:])), axis=0)
                tp_2022 = np.concatenate((tp_2022, np.array(nc_obj_2022.variables['tp'][:])), axis=0)
        print('气象 数据中的时间(2023):', time_2022.shape)

        # ---- 读取 2024 气象（逐月拼接）----
        for i in range(1, 13):
            file_path = f'/home/24meteroloy_data/2024{str(i).zfill(2)}.nc'
            nc_obj = Dataset(file_path)
            if 'time' in nc_obj.variables:
                current_time = np.array(nc_obj.variables['time'][:])
            elif 'valid_time' in nc_obj.variables:
                current_time = np.array(nc_obj.variables['valid_time'][:])
            else:
                raise KeyError("No time variable found in file: " + file_path)

            if i == 1:
                longitude = np.array(nc_obj.variables['longitude'][:])
                latitude = np.array(nc_obj.variables['latitude'][:])
                time = current_time
                t2m = np.concatenate((t2m_2022, np.array(nc_obj.variables['t2m'][:])), axis=0)
                u10 = np.concatenate((u10_2022, np.array(nc_obj.variables['u10'][:])), axis=0)
                v10 = np.concatenate((v10_2022, np.array(nc_obj.variables['v10'][:])), axis=0)
                d2m = np.concatenate((d2m_2022, np.array(nc_obj.variables['d2m'][:])), axis=0)
                sp = np.concatenate((sp_2022, np.array(nc_obj.variables['sp'][:])), axis=0)
                tp = np.concatenate((tp_2022, np.array(nc_obj.variables['tp'][:])), axis=0)
            else:
                time = np.concatenate((time, current_time), axis=0)
                t2m = np.concatenate((t2m, np.array(nc_obj.variables['t2m'][:])), axis=0)
                u10 = np.concatenate((u10, np.array(nc_obj.variables['u10'][:])), axis=0)
                v10 = np.concatenate((v10, np.array(nc_obj.variables['v10'][:])), axis=0)
                d2m = np.concatenate((d2m, np.array(nc_obj.variables['d2m'][:])), axis=0)
                sp = np.concatenate((sp, np.array(nc_obj.variables['sp'][:])), axis=0)
                tp = np.concatenate((tp, np.array(nc_obj.variables['tp'][:])), axis=0)

        print('加载完气象数据之后的维度：', t2m.shape)
        print('气象 数据中的时间(2023+2024)：', time.shape)

        # ---- 加载污染物数据（用循环避免冗余）----
        pol_2023_dir = '/home/2023pollute_data'
        pol_2024_dir = '/home/2024pollute_data'
        stations = [
            '1001A','1003A','1004A','1005A','1006A','1007A','1008A','1009A','1010A','1011A',
            '1012A','1015A','1017A','1018A','1019A','1021A','1024A','1029A','1030A','1031A',
            '1032A','1033A','1034A','1037A','1040A','1041A','1042A','1043A','1046A','1051A',
            '1052A','1053A','1054A','1057A','1059A','1061A','1062A','1063A','1064A','1065A',
            '1066A','1067A','1068A','1070A','1071A','1072A','1073A','1074A','1075A','1077A',
            '1078A','1079A','1080A','2859A','2860A','2862A','2919A','2922A','3020A','3051A',
            '3131A','3188A','3281A','3323A','3324A','3325A','3327A','3417A','3418A','3456A',
            '3457A','3459A','3460A','3461A','3462A','3573A','3574A','3575A','3576A','3577A',
            '3578A','3579A','3580A','3581A','3582A','3583A','3672A','3673A','3674A','3675A',
            '3692A','3693A','3694A','3695A','3697A'
        ]

        def _read_and_interp(dirpath, name):
            df = pd.read_csv(f'{dirpath}/2024_{name}.csv') if '2024' in dirpath else pd.read_csv(f'{dirpath}/2023_{name}.csv')
            df = df[stations].interpolate(method='linear', limit_direction='forward')
            df = df.interpolate(method='linear', limit_direction='backward')
            return df

        pol_names = ['AQI', 'CO', 'NO2', 'O3', 'PM10', 'PM25', 'SO2']
        # 2023 & 2024 两年数据一并处理
        pol_2023 = {n: _read_and_interp(pol_2023_dir, n) for n in pol_names}
        pol_2024 = {n: _read_and_interp(pol_2024_dir, n) for n in pol_names}

        # 拼接为整年 → ndarray
        pol_all = {n: np.concatenate([pol_2023[n].to_numpy(), pol_2024[n].to_numpy()], axis=0) for n in pol_names}

        # ---- 加载坐标数据 ----
        dstation = pd.read_excel('/home/station.xlsx')
        coordinates = np.array(dstation[['latitude', 'longitude']]).reshape(-1, 2)
        print('coordinates.shape:', coordinates.shape)

        # ---- 将气象格点映射到各站 (最近格点) ----
        dt2m = np.zeros((t2m.shape[0], coordinates.shape[0]))
        du10 = np.zeros_like(dt2m)
        dv10 = np.zeros_like(dt2m)
        dd2m = np.zeros_like(dt2m)
        dsp  = np.zeros_like(dt2m)
        dtp  = np.zeros_like(dt2m)

        wlongitude = [np.argmin(np.abs(longitude - coordinates[i, 1])) for i in range(coordinates.shape[0])]
        wlatitude  = [np.argmin(np.abs(latitude  - coordinates[i, 0])) for i in range(coordinates.shape[0])]
        print('wlatitude/longitude 计算完成')

        wlongitude = np.array(wlongitude).reshape(coordinates.shape[0])
        wlatitude  = np.array(wlatitude).reshape(coordinates.shape[0])

        for k in range(t2m.shape[0]):
            for j in range(coordinates.shape[0]):
                dt2m[k][j] = t2m[k][wlatitude[j]][wlongitude[j]]
                du10[k][j] = u10[k][wlatitude[j]][wlongitude[j]]
                dv10[k][j] = v10[k][wlatitude[j]][wlongitude[j]]
                dd2m[k][j] = d2m[k][wlatitude[j]][wlongitude[j]]
                dsp[k][j]  = sp[k][wlatitude[j]][wlongitude[j]]
                dtp[k][j]  = tp[k][wlatitude[j]][wlongitude[j]]

        # 填补缺失
        dt2m = self._fill_missing(dt2m); du10 = self._fill_missing(du10); dv10 = self._fill_missing(dv10)
        dd2m = self._fill_missing(dd2m); dsp  = self._fill_missing(dsp);  dtp  = self._fill_missing(dtp)

        print('整理完对应的站点的气象数据之后的维度：', dt2m.shape)
        daqi  = pol_all['AQI']
        dco   = pol_all['CO']
        dno2  = pol_all['NO2']
        do3   = np.nan_to_num(pol_all['O3'])
        dpm10 = pol_all['PM10']
        dpm25 = pol_all['PM25']
        dso2  = pol_all['SO2']

        num_features = 13
        data = np.zeros((daqi.shape[0], 1))
        for i in range(coordinates.shape[0]):
            data = np.concatenate((data,
                                   dt2m[:, i].reshape(-1, 1),
                                   du10[:, i].reshape(-1, 1), dv10[:, i].reshape(-1, 1),
                                   dd2m[:, i].reshape(-1, 1),
                                   dsp[:, i].reshape(-1, 1),
                                   dtp[:, i].reshape(-1, 1),
                                   daqi[:, i].reshape(-1, 1),
                                   dco[:, i].reshape(-1, 1),
                                   dno2[:, i].reshape(-1, 1),
                                   do3[:, i].reshape(-1, 1),
                                   dpm10[:, i].reshape(-1, 1),
                                   dso2[:, i].reshape(-1, 1),
                                   dpm25[:, i].reshape(-1, 1)), axis=1)

        data = np.delete(data, obj=0, axis=1)
        data_final = np.zeros((coordinates.shape[0], daqi.shape[0], num_features))
        for i in range(coordinates.shape[0]):
            data_final[i] = data[:, i * num_features:(i + 1) * num_features]

        X = data_final
        data_normal = []
        num_nodes = data_final.shape[0]
        features = data_final.shape[2]

        Y = data_final[:, :, -1].reshape(-1, 1)  # PM2.5（按你原逻辑取最后一列）
        for i in range(data_final.shape[0]):
            data_temp, _, _ = self._Maxmin(data_final[i, :, :])
            data_normal.append(data_temp)

        data = np.concatenate((data_normal), axis=0)
        Y = Y.reshape(num_nodes, -1).T
        data = data.reshape(num_nodes, -1, features)
        X = data.transpose(1, 0, 2)

        print('最后X的维度：', X.shape, 'y的维度：', Y.shape)
        return X, Y, X.shape[0], X.shape[1], X.shape[2], coordinates, longitude, latitude,  # grey_corr_adj

    def _getAdj(self, k=4, sigma=None):
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

    def _Maxmin(self, data):
        maxdata = np.max(data, axis=0)
        mindata = np.min(data, axis=0)
        denominator = maxdata - mindata
        for i in range(denominator.shape[0]):
            if denominator[i] == 0:
                denominator[i] += 1e-8
        return (data - mindata) / denominator, maxdata, mindata

    def test_Maxmin(self, data, maxdata, mindata):
        return (data - mindata) / (maxdata - mindata)

    def deMaxmin(self, data, maxdata, mindata):
        return (data * (maxdata - mindata)) + mindata

    def _split(self, train, valid):
        print('trian:', train, 'valid:', valid)
        train_set = range(self.input_seq_len + self.output_seq_len - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.num)

        print('len_train_set:', len(train_set), 'len_valide:', len(valid_set), 'len_test:', len(test_set))

        self.train = self._batchify(train_set, self.output_seq_len)
        self.valid = self._batchify(valid_set, self.output_seq_len)
        self.test = self._batchify(test_set, self.output_seq_len)

    def _batchify(self, idx_set, horizon):
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

        print('分成批次之后的样本的形状：', dataX.shape, dataY.shape)
        return [dataX, dataY]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
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
    print(Data.X.shape, Data.Y.shape)
