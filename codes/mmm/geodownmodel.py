import numpy as np
import torch
import torch.nn as nn
# from .datahandler import DataHandler
from .gcnmodel import GCNLayer
from .geobasemodel import BlockRecognizer
from .mymodel import MyBaseModel
from torch.autograd import Variable
from tqdm import tqdm


class GCNModel(nn.Module):
    def __init__(self, category, rep_category, relationship, rep_weight,
                 base_weight, BR_settings={}):
        '''
        コンストラクタ
        '''
        super().__init__()
        self._use_gpu = torch.cuda.is_available()
        self.num_classes = len(category)

        # H_0の作成
        fc_weight = np.array(rep_weight['fc.weight'].cpu(), dtype=np.float64)
        H_0 = np.zeros((self.num_classes, 100))
        for label, index in rep_category.items():
            H_0[category[label]] = fc_weight[index]

        # Aの作成
        A = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for label, _ in category.items():
            if label not in relationship:
                continue

            children = relationship[label]
            A[category[label]][category[label]] = 1
            for child in children:
                if child in category:
                    A[category[label]][category[child]] = 1

        # 特徴抽出の部分を定義
        self._before_fc = BlockRecognizer(**BR_settings)
        self._before_fc.load_state_dict(base_weight)
        self._mean = self._before_fc._mean
        self._std = self._before_fc._std
        self._before_fc = torch.nn.Sequential(*(
            list(self._before_fc.children())[:-2]
        ))

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
        output = output.view(1, 100)

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

    def _zero_weight(self, labels):
        weight = np.ones(labels.shape, int) * self._backprop_weight[0]
        weight = torch.Tensor(weight).cuda() if self._use_gpu else weight

        return weight

    def _processing_one_epoch(self, mode, dataset, epoch):
        '''
        1 epochで行う処理の記述

        return:
            tuple (loss, recall, precision, filenames, predicts, labels)
            filenames: トレーニングに用いた画像のファイル名．
            predict, labels: filenameに対応するラベルの予測値と正解ラベル
        '''
        running_loss, correct, total, pred_total = 0, 0, 0, 0

        for i, (images, labels, filename) in enumerate(tqdm(dataset)):
            if self._use_gpu:
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()

            outputs = self._model(images)
            predicted = np.array(outputs.cpu().data) >= 0

            if labels.sum() == 0:
                batch_loss = self._loss_function(
                    outputs, labels,
                    self._zero_weight(labels)
                )
            else:
                # lossの計算
                batch_loss = self._loss_function(
                    outputs, labels, self._update_backprop_weight(labels)
                )

                # lossとかrecallとかprecisionとかを出すために必要な値を計算
                running_loss += float(batch_loss)
                correct += np.sum(np.logical_and(
                    np.array(labels.cpu().data), predicted)
                )
                total += labels.data.sum()
                pred_total += predicted.sum()

            # modeが'train'の時はbackprop
            if mode == 'train':
                self._optimizer.zero_grad()
                batch_loss.backward()
                self._optimizer.step()

        epoch_loss = running_loss / len(dataset)
        recall = correct / total
        precision = 0.0 if pred_total == 0 else correct / pred_total

        return epoch_loss, recall, precision
