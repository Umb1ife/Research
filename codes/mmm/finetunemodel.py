import itertools
import torch.nn as nn
from .mymodel import MyBaseModel
from torchvision import models


class FinetuneModel(MyBaseModel):
    '''
    Fine-tuningで学習する識別器
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _mynetwork(self, _):
        '''
        学習するネットワークの定義．
        訓練済みresnet101をベースにFine-tuningを行う
        '''
        self._model = models.resnet101(pretrained=True)
        for param in self._model.parameters():
            param.requires_grad = False

        self._model.fc = nn.Linear(self._model.fc.in_features, self._class_num)

        # backprop settings
        bp_list = [
            self._model.layer4[1].parameters(),
            self._model.layer4[2].parameters(),
            self._model.fc.parameters()
        ]

        for lyr in bp_list:
            for param in lyr:
                param.requires_grad = True

        # optimizerに渡すレイヤーのパラメータを取得
        bp_list = [
            list(self._model.layer4[1].parameters()),
            list(self._model.layer4[2].parameters()),
            list(self._model.fc.parameters())
        ]
        self._optimizer = self._optimizer(
            itertools.chain(*bp_list), lr=self._lr, momentum=self._momentum
        )
