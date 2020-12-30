# import json
import numpy as np
import torch
import torch.nn as nn
from .datahandler import DataHandler
from .gcnmodel import GCNLayer
from .georepmodel import SimpleGeoNet
from .mymodel import MyBaseModel


class GCNModel(nn.Module):
    def __init__(self, category, rep_category, filepaths={},
                 feature_dimension=30, simplegeonet_settings={}):
        '''
        コンストラクタ
        '''
        super().__init__()
        self._use_gpu = torch.cuda.is_available()
        self._feature_dimension = feature_dimension
        self.num_classes = len(category)

        relationship = DataHandler.loadPickle(filepaths['relationship'])
        CNN_weight = torch.load(filepaths['learned_weight'])

        # H_0の作成
        fc_weight = np.array(CNN_weight['fc3.weight'].cpu(), dtype=np.float64)
        H_0 = np.zeros((self.num_classes, feature_dimension))
        for label, index in rep_category.items():
            H_0[category[label]] = fc_weight[index]

        # Aの作成
        A = np.zeros((self.num_classes, self.num_classes), dtype=int)
        all_category_labels = list(category.keys())

        # for label, _ in upper_category.items():
        for label, _ in category.items():
            if label not in relationship:
                continue

            children = relationship[label]
            A[category[label]][category[label]] = 1
            for child in children:
                if child in all_category_labels:
                    A[category[label]][category[child]] = 1

        # GeoRepModelの最終層前までを取得
        self._before_fc = SimpleGeoNet(**simplegeonet_settings)
        self._mean = self._before_fc._mean
        self._std = self._before_fc._std
        self._before_fc.load_state_dict(CNN_weight)
        self._before_fc = torch.nn.Sequential(
            *(list(self._before_fc.children())[:-1])
        )

        # モデルの定義
        # 層を追加するときは下のfowardも変更
        self.layer1 = GCNLayer(H_0, A)

    def forward(self, inputs):
        '''
        入力をネットワークに通したときの出力(クラスの予測)
        '''
        feature = torch.autograd.Variable(inputs).float()
        feature = self._pass_before_fc(feature)
        output = self.layer1(feature)

        return output

    def _pass_before_fc(self, inputs):
        '''
        Fine-tuningでのFC層前までの特徴量を計算
        '''
        self._before_fc.eval()
        if self._use_gpu:
            inputs = inputs.cuda()

        inputs = (inputs - self._mean) / self._std
        output = self._before_fc(inputs)
        output = output.view(1, self._feature_dimension)

        return output

    def __repr__(self):
        return


class GeotagGCN(MyBaseModel):
    '''
    GCNで学習する識別器
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _mynetwork(self, settings):
        '''
        学習するネットワークの定義．
        '''
        self._model = GCNModel(**settings)
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.layer1.W.requires_grad = True

        self._optimizer = self._optimizer(
            self._model.layer1.parameters(),
            lr=self._lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay
        )
