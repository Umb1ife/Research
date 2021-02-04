import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from mmm import DataHandler as DH
from mmm import DatasetFlickr
from mmm.mymodel import MyBaseModel
from torchvision import models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


class PreviousMethod(MyBaseModel):
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
        self._optimizer = self._optimizer(
            self._model.parameters(), lr=self._lr, momentum=self._momentum
        )


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 初期設定
    # -------------------------------------------------------------------------
    batch_size = 16
    device_ids = '0, 1, 2, 3'
    epochs = 200
    learning_rate = 0.1
    sim_threshold = 0.4
    workers = 4
    input_path = '../datas/gcn/inputs/'

    # パラメータや使用するGPUあたりの設定
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids

    batchsize = batch_size * len(device_ids.split(','))
    print('Number of GPUs: {}'.format(len(device_ids.split(','))))
    numwork = workers

    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    kwargs_DF = {
        'train': {
            'filenames': {
                'Annotation': input_path + 'train_anno.json',
                'Category_to_Index': input_path + 'category.json'
            },
            'transform': transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(1.0, 1.0), ratio=(1.0, 1.0)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=image_normalization_mean,
                        std=image_normalization_std
                    )
                ]
            ),
            'image_path': input_path + 'images/train/'
        },
        'validate': {
            'filenames': {
                'Annotation': input_path + 'validate_anno.json',
                'Category_to_Index': input_path + 'category.json'
            },
            'transform': transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(1.0, 1.0), ratio=(1.0, 1.0)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=image_normalization_mean,
                        std=image_normalization_std
                    )
                ]
            ),
            'image_path': input_path + 'images/validate/'
        }
    }

    train_dataset = DatasetFlickr(**kwargs_DF['train'])
    val_dataset = DatasetFlickr(**kwargs_DF['validate'])
    num_class = train_dataset.num_category()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batchsize,
        num_workers=numwork
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=batchsize,
        num_workers=numwork
    )

    if torch.cuda.is_available():
        train_loader.pin_memory = True
        val_loader.pin_memory = True
        cudnn.benchmark = True

    # maskの読み込み
    mask = DH.loadPickle(
        '{0:0=2}'.format(int(sim_threshold * 10)),
        input_path + 'comb_mask/'
    )

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy(
        '{0:0=2}'.format(int(sim_threshold * 10)),
        input_path + 'backprop_weight/'
    )

    # 何も工夫なしにfine-tuning、マスク有り、マスク＋weight有りで試す？
    model = PreviousMethod(
        class_num=num_class,
        loss_function=MyLossFunction(),
        optimizer=optim.SGD,
        learningrate=learning_rate,
        momentum=0.9,
        fix_mask=mask,
        multigpu=True if len(device_ids.split(',')) > 1 else False,
        backprop_weight=bp_weight
    )
    # -------------------------------------------------------------------------
    # 学習
    # -------------------------------------------------------------------------
    writer = SummaryWriter(log_dir='../datas/comparison/mask_and_bp/')

    train_loss, train_recall, train_precision = model.validate(train_loader)
    val_loss, val_recall, val_precision = model.validate(val_loader)
    print('epoch: {0}'.format(0))
    print('loss: {0}, recall: {1}, precision: {2}'.format(
        train_loss, train_recall, train_precision
    ))
    print('loss: {0}, recall: {1}, precision: {2}'.format(
        val_loss, val_recall, val_precision
    ))
    writer.add_scalar('loss', train_loss, 0)
    writer.add_scalar('recall', train_recall, 0)
    writer.add_scalar('precision', train_precision, 0)
    print('------------------------------------------------------------------')

    for epoch in range(1, epochs + 1):
        train_loss, train_recall, train_precision = model.train(train_loader)
        val_loss, val_recall, val_precision = model.validate(val_loader)
        print('epoch: {0}'.format(epoch))
        print('loss: {0}, recall: {1}, precision: {2}'.format(
            train_loss, train_recall, train_precision
        ))
        print('loss: {0}, recall: {1}, precision: {2}'.format(
            val_loss, val_recall, val_precision
        ))
        print('--------------------------------------------------------------')

        writer.add_scalar('loss', train_loss, epoch)
        writer.add_scalar('recall', train_recall, epoch)
        writer.add_scalar('precision', train_precision, epoch)

    print('finish.')
