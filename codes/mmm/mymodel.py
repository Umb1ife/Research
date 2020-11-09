import numpy as np
import os
import torch
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from .customized_multilabel_soft_margin_loss \
    import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from torch.autograd import Variable
from tqdm import tqdm


class MyBaseModel(metaclass=ABCMeta):
    '''
    学習を行うモデルのスーパークラス．
    このクラスそのものはインスタンス化できない
    '''
    def __init__(self, *, class_num=0,
                 loss_function=MyLossFunction(),
                 optimizer=optim.SGD, learningrate=0.01, momentum=0.9,
                 weight_decay=None, fix_mask=None, multigpu=False,
                 backprop_weight=None, network_setting={}):
        '''
        コンストラクタ

        Arguments:
            class_num: 学習する分類器が持つクラス数
            fix_mask: 誤差を伝播させない部分を指定するマスク
            network_setting: オーバーライドした_mynetworkメソッドで必要な場合は指定
        '''
        self._class_num = class_num
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._lr = learningrate
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._fix_mask = np.zeros((class_num, class_num), dtype=int) \
            if fix_mask is None else fix_mask
        self._multigpu = multigpu
        self._use_gpu = torch.cuda.is_available()
        self._activation_function = np.vectorize(self._activation_function)

        # backprop_weightをデータ数の逆数で補正
        backprop_weight = np.ones((class_num, 2)) \
            if backprop_weight is None else backprop_weight
        backprop_weight = np.sqrt(backprop_weight)
        backprop_weight[backprop_weight == 0] = np.inf
        self._backprop_weight = 1 / backprop_weight

        self._backprop_weight = np.array(self._backprop_weight).T

        # ネットワークの定義
        self._mynetwork(network_setting)

        # GPUを使うために変換
        if self._use_gpu:
            self._model = self._model.cuda()
            self._loss_function = self._loss_function.cuda()

        # 複数GPU使うときの設定
        if self._multigpu:
            self._model = torch.nn.DataParallel(self._model)
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True

    @abstractmethod
    def _mynetwork(self, kwargs):
        '''
        学習するモデルや更新する(しない)レイヤーの設定など
        '''

    @staticmethod
    def _activation_function(x):
        '''
        活性化関数の定義．ここではSigmoid関数
        '''
        sigmoid_range = 34.538776394910684
        if x <= -sigmoid_range:
            return 1e-15
        elif x >= sigmoid_range:
            return 1.0 - 1e-15
        else:
            return 1.0 / (1.0 + np.exp(-x))

    def _update_backprop_weight(self, labels):
        '''
        誤差を伝播させる際の重みを指定．誤差を伝播させない部分は0．
        '''
        labels = labels.data.cpu().numpy()
        labels_y, labels_x = np.where(labels == 1)
        labels_y = np.append(labels_y, labels_y[-1] + 1)
        labels_x = np.append(labels_x, 0)

        weight = np.zeros((labels.shape[0] + 1, labels.shape[1]), int)
        row_p, columns_p = labels_y[0], [labels_x[0]]
        weight[row_p] = self._fix_mask[labels_x[0]]

        for row, column in zip(labels_y[1:], labels_x[1:]):
            if row == row_p:
                columns_p.append(column)
            else:
                if len(columns_p) > 1:
                    for y in columns_p:
                        weight[row_p][y] = 0

                row_p, columns_p = row, [column]

            weight[row] = weight[row] | self._fix_mask[column]

        weight = weight[:-1]
        weight = np.ones(labels.shape, int) - weight
        labels_zero = np.ones(labels.shape, int) - labels

        weight = weight * labels_zero * self._backprop_weight[0] \
            + weight * labels * self._backprop_weight[1]
        weight = torch.Tensor(weight).cuda() if self._use_gpu else weight

        return weight

    def _predict(self, outputs, th=0.5):
        '''
        活性化関数後の値について，thを基準に予測クラスを1に，それ以外を0に変更
        '''
        outputs = outputs.data.cpu().numpy()
        outputs = self._activation_function(outputs)
        outputs[outputs >= th] = 1
        outputs[outputs < th] = 0

        return outputs

    def _processing_one_epoch(self, mode, dataset, epoch):
        '''
        1 epochで行う処理の記述

        return:
            tuple (loss, recall, precision, filenames, predicts, labels)
            filenames: トレーニングに用いた画像のファイル名．
            predict, labels: filenameに対応するラベルの予測値と正解ラベル
        '''
        predict_list, ans_list = [], []
        running_loss, correct, total, pred_total = 0, 0, 0, 0

        filelist = []
        for i, (images, labels, filename) in enumerate(tqdm(dataset)):
            filelist.append(filename)
            if self._use_gpu:
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()

            outputs = self._model(images)
            predicted = np.array(outputs.cpu().data) >= 0

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

            # 予想結果と正解を保存
            predict_list.append(self._predict(outputs))
            ans_list.append(labels.data.cpu().numpy())

            # modeが'train'の時はbackprop
            if mode == 'train':
                self._optimizer.zero_grad()
                batch_loss.backward()
                self._optimizer.step()

        epoch_loss = running_loss / len(dataset)
        recall = correct / total
        precision = 0.0 if pred_total == 0 else correct / pred_total

        return epoch_loss, recall, precision, filelist, predict_list, ans_list

    def train(self, dataset, epoch=None):
        '''
        引数datasetに渡されたデータをもとに自身が持つモデルをトレーニング

        return:
            tuple (loss, recall, precision, filenames, predicts, labels)
        '''
        # training modeへ
        self._model.train()

        return self._processing_one_epoch(
            mode='train', dataset=dataset, epoch=epoch
        )

    def validate(self, dataset, epoch=None):
        '''
        引数datasetに渡されたデータを用いて，現在自身が持っているモデルでvalidate．

        return:
            tuple (loss, recall, precision, filenames, predicts, labels)
        '''
        # evaluate modeへ
        self._model.eval()

        return self._processing_one_epoch(
            mode='validate', dataset=dataset, epoch=epoch
        )

    def loadmodel(self, filename, path=None):
        '''
        保存したモデルのパラメータをload．拡張子: .pth
        '''
        filename = str(filename)
        filename = filename if filename[-1:-5:-1] == 'htp.' \
            else filename + '.pth'
        if path is not None:
            path = path if path[-1:] == '/' else path + '/'
            filename = path + filename

        if self._multigpu:
            self._model.module.load_state_dict(torch.load(filename))
        else:
            self._model.load_state_dict(torch.load(filename))

    def savemodel(self, filename, path=None):
        '''
        現在のmodelのパラメータを保存．拡張子: .pth
        '''
        filename = str(filename)
        filename = filename if filename[-1:-5:-1] == 'htp.' \
            else filename + '.pth'
        if path is not None:
            path = path if path[-1:] == '/' else path + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            filename = path + filename

        if self._multigpu:
            torch.save(self._model.module.state_dict(), filename)
        else:
            torch.save(self._model.state_dict(), filename)

    def predict(self, testdata, normalized=True, labeling=False, th=0.5):
        if self._use_gpu:
            testdata = Variable(testdata).cuda()

        self._model.eval()
        output = self._model(testdata)
        if normalized:
            output = output.data.cpu().numpy()
            output = self._activation_function(output)
            if labeling:
                output[output >= th] = 1
                output[output < th] = 0

        return output
