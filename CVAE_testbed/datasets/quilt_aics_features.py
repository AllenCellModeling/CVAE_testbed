import torch
from quilt3 import Package
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
import os
import json
from featuredb import FeatureDatabase
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA


class QuiltAicsFeatures(Dataset):
    def __init__(
        self,
        num_batches,
        BATCH_SIZE,
        model_kwargs,
        shuffle=True,
        corr=False,
        train=True,
        mask=False
                ):
        """
        Args:
            num_batches: Number of batches of synthetic data
            BATCH_SIZE: batchsize of synthetic data
            model_kwargs: dictionary containing "x_dim"
            which indicates input data size
            shuffle:  True sets condition vector in input data to 0
            for all possible permutations
            corr: True sets dependent input dimensions
            via a correlation matrix
        """
        self.num_batches = num_batches
        self.BATCH_SIZE = BATCH_SIZE
        self.corr = corr
        self.shuffle = shuffle
        self.model_kwargs = model_kwargs
        self.train = train

        Batches_C_train, Batches_C_test = torch.empty([0]), torch.empty([0])
        Batches_X_train, Batches_X_test = torch.empty([0]), torch.empty([0])
        Batches_conds_train, Batches_conds_test = torch.empty([0]), torch.empty([0])

        ds = Package.browse(
            "aics/pipeline_integrated_single_cell",
            "s3://allencell"
            )

        # Specify path to pre downloaded quilt json files
        try:
            path_to_json = model_kwargs['json_quilt_path']
        except:
            path_to_json = "/home/ritvik.vasan/test/"

        # json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

        meta_to_file_name = []
        for f in ds["cell_features"]:
            meta_to_file_name.append(
                {
                    "filename": f,
                    **ds["cell_features"][f].meta
                }
                                    )

        metas = pd.DataFrame(meta_to_file_name)

        # Specify path to config file for FeatureDatabase
        try:
            db = FeatureDatabase(model_kwargs['config_path'])
        except:
            db = FeatureDatabase("/home/ritvik.vasan/config.json")

        t = db.get_pg_table(
            "featuresets",
            "aics-mitosis-classifier-four-stage_v1.0.0"
            )

        semi = metas.merge(
            t,
            left_on="CellId",
            right_on="CellId",
            suffixes=("_meta", "_mito")
            )

        # Only interphase or no interphase
        semi['Interphase and Mitotic Stages [stage]'] = semi[
            'Interphase and Mitotic Stages [stage]'
            ].apply(lambda x: 0 if x == 0.0 else 1)

        dd = defaultdict(list)
        for i in range(len(semi['filename'])):
            this_file = semi['filename'][i]
            a = json.loads(open(path_to_json + this_file).read())
            a = dict(
                [
                    (key, value) for key, value in a.items()
                    if key not in [
                        'imsize_orig',
                        'com',
                        'angle',
                        'flipdim',
                        'imsize_registered'
                        ]
                ]
                    )
            a.update({'CellId': semi['CellId'][i]})
            for key, value in a.items():
                dd[key].append(value)

        features_plus_cellid = pd.DataFrame(dict(dd))

        meta_plus_features = pd.merge(
            semi,
            features_plus_cellid,
            on='CellId'
            )

        i_care_cols = [
            c for c in meta_plus_features.columns
            if c not in [
                'CellId',
                'CellIndex',
                'FOVId',
                'WellId',
                'FeatureExplorerURL',
                'CellLine',
                'Workflow',
                'associates',
                'filename',
                'NucMembSegmentationAlgorithm',
                'NucMembSegmentationAlgorithmVersion',
                'PlateId'
                        ]
                    ]

        meta_plus_features = meta_plus_features[i_care_cols]
        meta_plus_features.dropna(inplace=True)

        categorical_features = [
            'Gene',
            'ProteinDisplayName',
            'StructureDisplayName'
            ]

        categorical_dataframe = meta_plus_features[categorical_features]

        non_categorical_dataframe = meta_plus_features[
            [
                c for c in meta_plus_features.columns
                if c not in categorical_features
            ]
            ]

        one_hot_categorical_features = pd.get_dummies(
            categorical_dataframe,
            prefix=None,
            drop_first=True
            )

        # num_of_cells = len(non_categorical_dataframe)

        # This is mean, std normalization
        non_categorical_dataframe = non_categorical_dataframe.iloc[:, :]

        print(non_categorical_dataframe.shape)

        self._feature_names = [
            i for i in non_categorical_dataframe.columns
            ] + [
                i for i in one_hot_categorical_features.columns
                ]

        x = non_categorical_dataframe.values
        std_scaler = preprocessing.StandardScaler()
        # 0 is binary, dont scale that column
        x_train_and_test_scaled = std_scaler.fit_transform(
            x[:, 1:model_kwargs["x_dim"]+1]
            )
        x_train_scaled = std_scaler.fit_transform(
            x[:30000, 1:model_kwargs["x_dim"]+1]
            )
        x_test_scaled = std_scaler.transform(
            x[30000:, 1:model_kwargs["x_dim"]+1]
            )

        if model_kwargs["x_dim"] > 103:
            non_categorical_train = pd.DataFrame(
                np.concatenate((x[:30000, 0:1], x_train_scaled), axis=1)
                )
            non_categorical_test = pd.DataFrame(
                np.concatenate((x[30000:, 0:1], x_test_scaled), axis=1)
                )
            non_categorical_train_and_test = pd.DataFrame(
                np.concatenate((x[:, 0:1], x_train_and_test_scaled), axis=1)
                )
            # print(non_categorical_train.shape, non_categorical_test.shape)
            # print(len(self._feature_names))
            # print(non_categorical_train_and_test.shape)
            non_categorical_train_and_test.columns = self._feature_names[:103]
        else:
            non_categorical_train = pd.DataFrame(x_train_scaled)
            non_categorical_test = pd.DataFrame(x_test_scaled)
            non_categorical_train_and_test = pd.DataFrame(
                x_train_and_test_scaled
                )
            self._feature_names = self._feature_names[
                1:model_kwargs['x_dim']+1
                ]
            non_categorical_train_and_test.columns = self._feature_names[:]
        # print(non_categorical_train.shape, non_categorical_test.shape, len(self._feature_names))

        # Convert to torch tensor
        self._non_categorical_dataframe = non_categorical_train_and_test
        self._categorical_dataframe = one_hot_categorical_features

        X_train_whole_batch = torch.from_numpy(
            non_categorical_train.values
            ).float()
        X_test_whole_batch = torch.from_numpy(
            non_categorical_test.values
            ).float()
        all_categorical_X = torch.from_numpy(
            one_hot_categorical_features.values
            ).float()

        if model_kwargs["x_dim"] > 103:
            X_train_whole_batch = torch.cat(
                (X_train_whole_batch, all_categorical_X[:30000, :]),
                1
                )
            X_test_whole_batch = torch.cat(
                (X_test_whole_batch, all_categorical_X[30000:, :]),
                1
                )

        for j, i in enumerate(range(self.num_batches)):
            X_train = X_train_whole_batch[
                i*self.BATCH_SIZE: (i+1)*self.BATCH_SIZE,
                :
                ]
            X_test = X_test_whole_batch[
                i*self.BATCH_SIZE: (i+1)*self.BATCH_SIZE,
                :
                ]

            if X_train.size()[0] != self.BATCH_SIZE:
                break

            print(X_train.size(), X_test.size())
            print(Batches_X_train.size(), Batches_X_test.size())

            self._color = X_train[:, 0]

            C_train = X_train.clone()
            C_test = X_test.clone()

            count = 0
            if self.shuffle is True:
                while count == 0:
                    C_mask_train = torch.zeros(C_train.shape).bernoulli_(0.5)
                    C_mask_test = torch.zeros(C_test.shape).bernoulli_(0.5)
                    count = 1
            else:
                C_mask_train = torch.zeros(C_train.shape).bernoulli_(0)
                C_mask_test = torch.zeros(C_test.shape).bernoulli_(0)

            C_train[C_mask_train.byte()] = 0
            C_train_indicator = C_mask_train == 0

            C_test[C_mask_test.byte()] = 0
            C_test_indicator = C_mask_test == 0

            C_train = torch.cat(
                [
                    C_train.float(),
                    C_train_indicator.float()
                ],
                1
                               )
            C_test = torch.cat(
                [
                    C_test.float(),
                    C_test_indicator.float()
                ],
                1
                               )

            X_train = X_train.view([1, -1, X_train.size()[-1]])
            X_test = X_test.view([1, -1, X_test.size()[-1]])
            C_train = C_train.view([1, -1, X_train.size()[-1]*2])
            C_test = C_test.view([1, -1, X_test.size()[-1]*2])

            # Sum up
            conds_train = C_train[:, :, X_train.size()[-1]:].sum(2)
            conds_test = C_test[:, :, X_test.size()[-1]:].sum(2)

            Batches_X_train = torch.cat([Batches_X_train, X_train], 0)
            Batches_C_train = torch.cat([Batches_C_train, C_train], 0)
            Batches_conds_train = torch.cat(
                [Batches_conds_train, conds_train],
                0
                )
            try:
                Batches_X_test = torch.cat([Batches_X_test, X_test], 0)
                Batches_C_test = torch.cat([Batches_C_test, C_test], 0)
                Batches_conds_test = torch.cat(
                    [Batches_conds_test, conds_test],
                    0
                    )
            except:
                pass

        self._batches_x_train = Batches_X_train
        self._batches_c_train = Batches_C_train
        self._batches_conds_train = Batches_conds_train

        self._batches_x_test = Batches_X_test
        self._batches_c_test = Batches_C_test
        self._batches_conds_test = Batches_conds_test

    def __len__(self):
        return len(self._batches_x_train)

    def __getitem__(self, idx):
        """
        Returns a tuple. (X, C, sum(C[mid:end])).
        X is the input,
        C is the condition,
        sum(C[mid:end]) is the sum of the indicators in C.
        It tells us how many of the condition
        columns have been masked
        """
        return self._batches_x_train[idx], self._batches_c_train[idx], self._batches_conds_train[idx]

    def get_train_data(self):
        return self._batches_x_train, self._batches_c_train, self._batches_conds_train

    def get_test_data(self):
        return self._batches_x_test, self._batches_c_test, self._batches_conds_test

    def get_dataframes(self):
        return self._non_categorical_dataframe, self._categorical_dataframe

    def get_feature_names(self):
        return self._feature_names

    def get_color(self):
        return self._color

    def get_all_items(self):
        if self.train is True:
            return self._batches_x_test, self._batches_c_test, self._batches_conds_test, None
        else:
            return self._batches_x_test, self._batches_c_test, self._batches_conds_test

    def get_PCA_features(self, min_variance):
        """
        Return the features that explain a minimum
        variance of the entire dataset
        """
        all_features = pd.merge(
                    self._non_categorical_dataframe,
                    self._categorical_dataframe,
                    on=self._non_categorical_dataframe.index
                               )
        all_features = all_features[
            [
                i for i in all_features.columns
                if i not in ['key_0']
            ]
                                   ]
        x_features = all_features.values

        for i in range(x_features.shape[-1]):
            model = PCA(n_components=i).fit(x_features)
            x_pc = model.transform(x_features)
            n_pcs = model.components_.shape[0]
            if np.sum(model.explained_variance_ratio_) > min_variance:
                break
      
        most_important = [
            np.abs(model.components_[i]).argmax()
            for i in range(n_pcs)
                         ]

        initial_features_names = all_features.columns.values

        most_important_names = [
            initial_features_names[most_important[i]]
            for i in range(n_pcs)
                               ]

        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

        df = pd.DataFrame(dic.items())

        return df
