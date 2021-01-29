import torch
import torch.nn as nn
from .geobasemodel import BlockRecognizer
from .mymodel import MyBaseModel


class SimpleGeoNet(nn.Module):
    def __init__(self, class_num=48, mean=(0, 0), std=(1, 1)):
        super().__init__()
        inter_n1 = 20
        inter_n2 = 30
        self.fc1 = nn.Linear(2, inter_n1)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(inter_n1, inter_n2)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(inter_n2, class_num)

        self._mean = torch.tensor(mean).float()
        std = std if 0 not in std else (1, 1)
        self._std = torch.tensor(std).float()

        if torch.cuda.is_available():
            self._mean = self._mean.cuda()
            self._std = self._std.cuda()

    def forward(self, x):
        x = (x - self._mean) / self._std
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.fc3(x)

        return x


class GeoRepNet(nn.Module):
    '''
    位置情報でクラス分類を行うネットワーク
    '''
    def __init__(self, num_classes, base_weight_path, BR_settings):
        super().__init__()
        self._use_gpu = torch.cuda.is_available()

        self._before_fc = BlockRecognizer(**BR_settings)
        self._before_fc.load_state_dict(torch.load(base_weight_path))
        self._mean = self._before_fc._mean
        self._std = self._before_fc._std
        self._before_fc = torch.nn.Sequential(*(
            list(self._before_fc.children())[:-2]
        ))

        self.fc = torch.nn.Linear(100, num_classes)

    def forward(self, inputs):
        feature = torch.autograd.Variable(inputs).float()
        feature = self._pass_before_fc(feature)
        output = self.fc(feature)

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
        # output = output.view(1, self._feature_dimension)
        output = output.view(1, 100)

        return output


class RepGeoClassifier(MyBaseModel):
    '''
    位置情報でクラス分類を行うネットワーク
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _mynetwork(self, settings):
        '''
        学習するネットワークの定義．
        '''
        self._model = GeoRepNet(**settings)
        self._model._mean.requires_grad = False
        self._model._std.requires_grad = False
        for param in self._model._before_fc.parameters():
            param.requires_grad = False

        self._optimizer = self._optimizer(
            self._model.parameters(), lr=self._lr, momentum=self._momentum
        )


class _RepGeoClassifier(MyBaseModel):
    '''
    位置情報でクラス分類を行うネットワーク
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _mynetwork(self, settings):
        '''
        学習するネットワークの定義．
        '''
        self._model = SimpleGeoNet(**settings)
        self._optimizer = self._optimizer(
            self._model.parameters(), lr=self._lr, momentum=self._momentum
        )
