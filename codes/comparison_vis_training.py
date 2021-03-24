import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from mmm import DataHandler as DH
from mmm import DatasetFlickr
from mmm import VisUtils as VU
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
    learning_rate = 1
    workers = 4
    input_path = '../datas/vis_down/inputs/'

    # パラメータや使用するGPUあたりの設定
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids

    batchsize = batch_size * len(device_ids.split(','))
    print('Number of GPUs: {}'.format(len(device_ids.split(','))))
    numwork = workers

    category = DH.loadJson('category.json', input_path)
    rep_category = DH.loadJson('upper_category.json', input_path)

    vis_down_train = VU.down_anno(category, rep_category, 'train')
    vis_down_validate = VU.down_anno(category, rep_category, 'validate')
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=(1.0, 1.0), ratio=(1.0, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    kwargs_DF = {
        'train': {
            'category': category,
            'annotations': vis_down_train,
            'transform': transform,
            'image_path': input_path + 'images/train/'
        },
        'validate': {
            'category': category,
            'annotations': vis_down_validate,
            'transform': transform,
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
    mask = VU.down_mask(rep_category, category, saved=False)

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy('backprop_weight.npy', input_path)

    # マスク有り、マスク＋weight有りで試す？
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
    save_path = 'lmask_bp'
    mpath = '../datas/comparison/{0}/learned'.format(save_path)
    writer = SummaryWriter(
        log_dir='../datas/comparison/{0}/log/'.format(save_path)
    )

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
    model.savemodel('000weight.pth', mpath)
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

        model.savemodel('{0:0=3}weight'.format(epoch), mpath)

    print('finish.')
