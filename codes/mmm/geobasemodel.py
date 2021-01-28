import numpy as np
import torch
import torch.nn as nn
from .mymodel import MyBaseModel
from torch.autograd import Variable
from tqdm import tqdm


class BlockRecognizer(nn.Module):
    def __init__(self, x_range=(-175, -64), y_range=(18, 71),
                 fineness=(20, 20), mean=(0, 0), std=(1, 1)):
        super().__init__()
        inter_n1 = fineness[0]
        inter_n2 = sum(fineness)
        self.layer1 = nn.Linear(2, inter_n1)
        self.sigmoid1 = nn.Sigmoid()
        self.layer2 = nn.Linear(inter_n1, inter_n2)
        self.sigmoid2 = nn.Sigmoid()
        self.layer3 = nn.Linear(inter_n2, 100)
        self.sigmoid3 = nn.Sigmoid()
        self.fc_layer = nn.Linear(100, np.prod(fineness))
        self.softmax = nn.LogSoftmax(dim=1)

        self._mean = torch.nn.Parameter(torch.tensor(mean).float())
        std = std if 0 not in std else (1, 1)
        self._std = torch.nn.Parameter(torch.tensor(std).float())

    def forward(self, x):
        x = (x - self._mean) / self._std
        x = self.layer1(x)
        x = self.sigmoid1(x)
        x = self.layer2(x)
        x = self.sigmoid2(x)
        x = self.layer3(x)
        x = self.sigmoid3(x)
        x = self.fc_layer(x)
        x = self.softmax(x)

        return x


class GeoBaseNet(MyBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _mynetwork(self, settings):
        '''
        学習するネットワークの定義
        '''
        self._model = BlockRecognizer(**settings)
        self._optimizer = self._optimizer(
            self._model.parameters(), lr=self._lr, momentum=self._momentum
        )

    @staticmethod
    def _activation_function(x):
        '''
        活性化関数の定義．Softmax関数
        '''
        x_max = max(x)
        x = np.exp(x - x_max)

        return x / np.sum(x)

    def _processing_one_epoch(self, mode, dataset, epoch):
        '''
        1 epochで行う処理の記述

        return:
            tuple (loss, recall, precision)
        '''
        running_loss = 0
        conf_mat = np.zeros((self._class_num, self._class_num))

        for locates, labels, _ in tqdm(dataset):
            if self._use_gpu:
                locates = Variable(locates).cuda()
                labels = Variable(labels).cuda()

            outputs = self._model(locates)
            predicted = np.argmax(outputs.cpu().data, axis=1)

            # lossの計算
            batch_loss = self._loss_function(outputs, labels)

            # lossとかrecallとかprecisionとかを出すために必要な値を計算
            running_loss += float(batch_loss)
            for label, pred in zip(labels, predicted):
                conf_mat[label][pred] += 1

            # modeが'train'の時はbackprop
            if mode == 'train':
                self._optimizer.zero_grad()
                batch_loss.backward()
                self._optimizer.step()

        epoch_loss = running_loss / len(dataset)
        diags = np.diag(conf_mat)
        rs, ps = np.sum(conf_mat, axis=1), np.sum(conf_mat, axis=0)
        rs, ps = np.where(rs == 0, 1, rs), np.where(ps == 0, 1, ps)
        recall, precision = np.mean(diags / rs), np.mean(diags / ps)

        return epoch_loss, recall, precision

    def predict(self, testdata, labeling=False):
        if self._use_gpu:
            testdata = Variable(testdata).cuda()

        self._model.eval()
        output = self._model(testdata)
        if labeling:
            output = np.argmax(output.cpu().data, axis=1)

        return output
