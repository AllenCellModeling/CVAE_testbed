import torch
from quilt3 import Package
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
import os, json
from featuredb import FeatureDatabase
from sklearn import preprocessing
import numpy as np
from scipy import stats
from scipy.stats import zscore
import sklearn


class QuiltAicsFeatures(Dataset):
    def __init__(self, num_batches, BATCH_SIZE, model_kwargs, shuffle=True, corr=False, train=True, mask=False):
        """
        Args: 
            num_batches: Number of batches of synthetic data
            BATCH_SIZE: batchsize of synthetic data
            model_kwargs: dictionary containing "x_dim" which indicates input data size
            shuffle:  True sets condition vector in input data to 0 for all possible permutations
            corr: True sets dependent input dimensions via a correlation matrix 
        """
        self.num_batches = num_batches
        self.BATCH_SIZE = BATCH_SIZE
        self.corr = corr
        self.shuffle = shuffle
        self.model_kwargs = model_kwargs
        self.train = train

        Batches_X_train, Batches_C_train, Batches_conds_train = torch.empty([0]), torch.empty([0]), torch.empty([0])
        Batches_X_test, Batches_C_test, Batches_conds_test = torch.empty([0]), torch.empty([0]), torch.empty([0])

        ds = Package.browse("aics/pipeline_integrated_single_cell", "s3://allencell")

        # Specify path to pre downloaded quilt json files
        try:
            path_to_json = model_kwargs['json_quilt_path']
        except:
            path_to_json = "/home/ritvik.vasan/test/"
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

        meta_to_file_name = []
        for f in ds["cell_features"]:
            meta_to_file_name.append({"filename": f, **ds["cell_features"][f].meta})
        metas = pd.DataFrame(meta_to_file_name)

        # Specify path to config file for FeatureDatabase
        try:
            db = FeatureDatabase(model_kwargs['config_path'])
        except:
            db = FeatureDatabase("/home/ritvik.vasan/config.json")

        t = db.get_pg_table("featuresets", "aics-mitosis-classifier-four-stage_v1.0.0")

        semi = metas.merge(t, left_on="CellId", right_on="CellId", suffixes=("_meta", "_mito"))

        # Only interphase or no interphase
        semi['Interphase and Mitotic Stages [stage]'] = semi['Interphase and Mitotic Stages [stage]'].apply(lambda x: 0 if x == 0.0 else 1)

        dd = defaultdict(list)
        for i in range(len(semi['filename'])):
            this_file = semi['filename'][i]
            a = json.loads(open('/home/ritvik.vasan/test/' + this_file).read())
            a = dict([(key, value) for key,value in a.items() if key not in ['imsize_orig', 'com', 'angle', 'flipdim', 'imsize_registered' ] ])
            a.update({'CellId': semi['CellId'][i]})
            for key, value in a.items():
                dd[key].append(value)

        features_plus_cellid = pd.DataFrame(dict(dd))

        meta_plus_features = pd.merge(semi, features_plus_cellid,on = 'CellId' )

        i_care_cols = [c for c in meta_plus_features.columns if c not in ['CellId',
                     'CellIndex', 'FOVId', 'WellId', 'FeatureExplorerURL',
                     'CellLine', 'Workflow','associates','filename', 
                     'NucMembSegmentationAlgorithm', 
                     'NucMembSegmentationAlgorithmVersion', 
                     'PlateId']]
        
        meta_plus_features = meta_plus_features[i_care_cols]
        meta_plus_features.dropna(inplace=True)

        categorical_features = ['Gene', 'ProteinDisplayName', 'StructureDisplayName']

        categorical_dataframe = meta_plus_features[categorical_features]

        non_categorical_dataframe = meta_plus_features[[c for c in meta_plus_features.columns if c not in categorical_features]]

        one_hot_categorical_features = pd.get_dummies(categorical_dataframe, prefix = None, drop_first = True)
        
        num_of_cells = len(non_categorical_dataframe)
        print(num_of_cells)

        # Here we normalize X with min max
        # x = non_categorical_dataframe.values
        # min_max_scaler = preprocessing.MinMaxScaler()
        # x_scaled = min_max_scaler.fit_transform(x)
        # non_categorical_dataframe = pd.DataFrame(x_scaled)

        # This is mean, std normalization
        non_categorical_dataframe = non_categorical_dataframe.iloc[:, 34:36]
        # non_categorical_dataframe[(np.abs(stats.zscore(non_categorical_dataframe) < 3)).all(axis=1)]
        # non_categorical_dataframe.apply(zscore)
        print(non_categorical_dataframe.shape)
        x = non_categorical_dataframe.values
        # x_train_scaled = x[:34000, :]
        # x_test_scaled = x[34000:, :]
        std_scaler = preprocessing.StandardScaler()
        x_train_scaled = std_scaler.fit_transform(x[:30000, :])
        x_test_scaled = std_scaler.transform(x[30000:, :])

        non_categorical_train = pd.DataFrame(x_train_scaled)
        non_categorical_test = pd.DataFrame(x_test_scaled)
        print(non_categorical_train.shape, non_categorical_test.shape )



        # x_train_scaled = sklearn.preprocessing.normalize(x[:28000, 1:6], axis=0)
        # x_test_scaled = sklearn.preprocessing.normalize(x[28000:, 1:6], axis=0)
        # non_categorical_train = pd.DataFrame(x_train_scaled)
        # non_categorical_test = pd.DataFrame(x_test_scaled)
        # print(non_categorical_train.shape, non_categorical_test.shape )
        # non_categorical_train = pd.DataFrame(np.concatenate((x[:28000, 0:1], x_train_scaled), axis = 1))
        # non_categorical_test = pd.DataFrame(np.concatenate((x[28000:, 0:1], x_test_scaled), axis = 1))
        
        # non_categorical_dataframe = non_categorical_dataframe.drop_duplicates()
        # non_categorical_dataframe = non_categorical_dataframe[(np.abs(stats.zscore(non_categorical_dataframe))<3).all(axis=1)]
        
        # Convert to torch tensor
        all_non_categorical_X_train = torch.from_numpy(non_categorical_train.values).float()
        all_non_categorical_X_test = torch.from_numpy(non_categorical_test.values).float()
        all_categorical_X = torch.from_numpy(one_hot_categorical_features.values).float()


        for j, i in enumerate(range(self.num_batches)):
            X_train = all_non_categorical_X_train[i*self.BATCH_SIZE: (i+1)*self.BATCH_SIZE, :]
            X_test = all_non_categorical_X_test[i*self.BATCH_SIZE: (i+1)*self.BATCH_SIZE, :]
            X_categorical = all_categorical_X[i*self.BATCH_SIZE: (i+1)*self.BATCH_SIZE, :]

            # X = torch.cat((X, X_categorical), 1)

            if X_train.size()[0] != self.BATCH_SIZE:
                break
            print(X_train.size(), X_test.size(), Batches_X_train.size(), Batches_X_test.size())

            self._color = X_train[:, 0]
        
            C_train = X_train.clone()
            C_test = X_test.clone()

            count = 0
            if self.shuffle is True:
                while count == 0:
                    C_mask_train = torch.zeros(C_train.shape).bernoulli_(0.5)
                    C_mask_test = torch.zeros(C_test.shape).bernoulli_(0.5)
                    count=1 
            else:
                C_mask_train = torch.zeros(C_train.shape).bernoulli_(0)
                C_mask_test = torch.zeros(C_test.shape).bernoulli_(0)

            C_train[C_mask_train.byte()] = 0
            C_train_indicator = C_mask_train == 0

            C_test[C_mask_test.byte()] = 0
            C_test_indicator = C_mask_test == 0
            print('Out')

            C_train = torch.cat([C_train.float(), C_train_indicator.float()], 1)
            C_test = torch.cat([C_test.float(), C_test_indicator.float()], 1)

            X_train = X_train.view([1, -1, X_train.size()[-1]])
            X_test = X_test.view([1, -1, X_test.size()[-1]])
            C_train = C_train.view([1, -1, X_train.size()[-1]*2])
            C_test = C_test.view([1, -1, X_test.size()[-1]*2])

            # Sum up
            conds_train = C_train[:, :, X_train.size()[-1]:].sum(2)
            conds_test = C_test[:, :, X_test.size()[-1]:].sum(2)

            Batches_X_train = torch.cat([Batches_X_train, X_train], 0)
            Batches_C_train = torch.cat([Batches_C_train, C_train], 0)
            Batches_conds_train = torch.cat([Batches_conds_train, conds_train], 0)
            try:
                Batches_X_test = torch.cat([Batches_X_test, X_test], 0)
                Batches_C_test = torch.cat([Batches_C_test, C_test], 0)
                Batches_conds_test = torch.cat([Batches_conds_test, conds_test], 0)
            except:
                pass

        self._batches_x_train = Batches_X_train
        self._batches_c_train = Batches_C_train
        self._batches_conds_train = Batches_conds_train

        self._batches_x_test = Batches_X_test
        self._batches_c_test = Batches_C_test
        self._batches_conds_test = Batches_conds_test
    
    def __len__(self):
        return len(self._batches_x)

    def __getitem__(self, idx):
        """
        Returns a tuple. (X, C, sum(C[mid:end])). 
        X is the input, 
        C is the condition, 
        sum(C[mid:end]) is the sum of the indicators in C. It tells us how many of the condition
        columns have been masked
        """
        return self._batches_x[idx], self._batches_c[idx], self._batches_conds[idx]

    def get_train_data(self):
        return self._batches_x_train, self._batches_c_train, self._batches_conds_train

    def get_test_data(self):
        return self._batches_x_test, self._batches_c_test, self._batches_conds_test
    
    def __len__(self):
        return len(self._batches_x)

    def __getitem__(self, idx):
        """
        Returns a tuple. (X, C, sum(C[mid:end])).
        X is the input, 
        C is the condition, 
        sum(C[mid:end]) is the sum of the indicators in C. It tells us how many of the condition
        columns have been masked
        """
        return self._batches_x[idx], self._batches_c[idx], self._batches_conds[idx]

    def get_color(self):
        return self._color

    def get_all_items(self):
        if self.train is True:
            return self._batches_x_test, self._batches_c_test, self._batches_conds_test, None
        else:
            return self._batches_x_test, self._batches_c_test, self._batches_conds_test