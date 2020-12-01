import json
import numpy as np
import torch
import torch.nn as nn
from .datahandler import DataHandler
from .gcnmodel import GCNLayer
from .georepmodel import SimpleGeoNet
from .mymodel import MyBaseModel
from torchvision import models


class GCNModel(nn.Module):
    def __init__(self, *, num_class=3100, filepaths={},
                 feature_dimension=2048):
        '''
        コンストラクタ
        '''
        super().__init__()
        self._use_gpu = torch.cuda.is_available()

        upper_category = json.load(open(filepaths['upper_category'], 'r'))
        category = json.load(open(filepaths['category'], 'r'))
        relationship = DataHandler.loadPickle(filepaths['relationship'])
        CNN_weight = torch.load(filepaths['learned_weight'])

        # H_0の作成
        fc_weight = np.array(CNN_weight['fc.weight'].cpu(), dtype=np.float64)
        H_0 = np.zeros((num_class, feature_dimension))
        for label, index in upper_category.items():
            H_0[category[label]] = fc_weight[index]

        # Aの作成
        A = np.zeros((num_class, num_class), dtype=int)
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

        # Fine-tuningしたresnetのFC層前までを取得
        self._before_fc = models.resnet101(pretrained=True)
        self._before_fc.fc = nn.Linear(
            self._before_fc.fc.in_features, len(upper_category)
        )
        self._before_fc.load_state_dict(CNN_weight)
        self._before_fc = torch.nn.Sequential(
            *(list(self._before_fc.children())[:-1])
        )

        self._before_fc = SimpleGeoNet()

        # モデルの定義
        # 層を追加するときは下のfowardも変更
        self.layer1 = GCNLayer(H_0, A)

    def forward(self, image):
        '''
        画像をネットワークに通したときの出力(クラスの予測)
        '''
        feature = torch.autograd.Variable(image).float()
        feature = self._pass_before_fc(feature)
        output = self.layer1(feature)

        return output

    def _pass_before_fc(self, image):
        '''
        Fine-tuningでのFC層前までの特徴量を計算
        '''
        self._before_fc.eval()
        if self._use_gpu:
            image = image.cuda()

        rimage = self._before_fc(image)
        rimage = rimage.view(image.shape[0], 2048)

        return rimage

    def __repr__(self):
        return


class MultiLabelGCN(MyBaseModel):
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
