import torch
import torch.nn as nn
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


class _SimpleGeoNet(nn.Module):
    def __init__(self, class_num=48, mean=(0, 0), std=(1, 1)):
        super().__init__()
        inter_n1 = 20
        inter_n2 = 30
        self.fc1 = nn.Linear(2, inter_n1)
        self.fc2 = nn.Linear(inter_n1, inter_n2)
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
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)

        return x


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
        self._model = SimpleGeoNet(**settings)
        self._optimizer = self._optimizer(
            self._model.parameters(), lr=self._lr, momentum=self._momentum
        )
