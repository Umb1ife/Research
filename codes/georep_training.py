import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from mmm import DataHandler as DH
from mmm import DatasetGeotag
from mmm import GeoUtils as GU
from mmm import MakeBPWeight
from mmm import RepGeoClassifier
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Fine-tuning')
parser.add_argument('--epochs', '-E', default=200, type=int, metavar='N')
parser.add_argument('--batch_size', '-B', default=1, type=int, metavar='N')
parser.add_argument('--load_mask', action='store_true')
parser.add_argument('--load_backprop_weight', action='store_true')
parser.add_argument(
    '--device_ids', '-D', default='0', type=str, metavar="'i, j, k'"
)
parser.add_argument(
    '--inputs_path', '-I', default='../datas/geo_rep/inputs', type=str,
    metavar='path of directory containing input data'
)
parser.add_argument(
    '--outputs_path', '-O', default='../datas/geo_rep/outputs/learned',
    type=str, metavar='path of directory trained model saved'
)
parser.add_argument(
    '--logdir', '-L', default='../datas/geo_rep/log', type=str,
    metavar='path of directory log saved'
)
parser.add_argument('--workers', '-W', default=4, type=int, metavar='N')
parser.add_argument(
    '--learning_rate', '-lr', default=0.1, type=float, metavar='N'
)
parser.add_argument('--start_epoch', default=1, type=int, metavar='N')


if __name__ == "__main__":
    args = parser.parse_args()

    # パラメータや使用するGPUあたりの設定
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    epochs = args.epochs
    batchsize = args.batch_size * len(args.device_ids.split(','))
    print('Number of GPUs: {}'.format(len(args.device_ids.split(','))))
    numwork = args.workers

    # データの読み込み先
    input_path = args.inputs_path if args.inputs_path[-1:] == '/' \
        else args.inputs_path + '/'
    category = DH.loadJson('category.json', input_path)
    num_class = len(category)

    # データの作成
    geo_rep_train, (mean, std) = GU.rep_dataset(category, 'train')
    geo_rep_validate, _ = GU.rep_dataset(category, 'validate')
    DH.savePickle(geo_rep_train, 'geo_rep_train', input_path)
    DH.savePickle(geo_rep_validate, 'geo_rep_validate', input_path)

    kwargs_DF = {
        'train': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data': geo_rep_train
        },
        'validate': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data': geo_rep_validate
        },
    }

    train_dataset = DatasetGeotag(**kwargs_DF['train'])
    val_dataset = DatasetGeotag(**kwargs_DF['validate'])

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
    mask = DH.loadPickle('mask_5', input_path) if args.load_mask else None
    mask = GU.rep_mask(category, reverse=False) if mask is None else mask

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy('backprop_weight', input_path) \
        if args.load_backprop_weight else None
    bp_weight = bp_weight if bp_weight is not None \
        else MakeBPWeight(train_dataset, num_class, mask, True, input_path)
    import numpy as np
    bp_weight = np.sqrt(bp_weight)

    # -------------------------------------------------------------------------
    geo_rep_train = GU.zerodata_augmentation(
        geo_rep_train, fineness=(50, 25), numdata_sqrt_oneclass=5
    )
    train_dataset = DatasetGeotag(**kwargs_DF['train'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batchsize,
        num_workers=numwork
    )

    if torch.cuda.is_available():
        train_loader.pin_memory = True
        cudnn.benchmark = True
    # -------------------------------------------------------------------------

    model = RepGeoClassifier(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        optimizer=optim.SGD,
        learningrate=args.learning_rate,
        momentum=0.9,
        fix_mask=mask,
        multigpu=True if len(args.device_ids.split(',')) > 1 else False,
        backprop_weight=bp_weight,
        network_setting={
            'num_classes': num_class,
            'base_weight_path': '../datas/geo_base/outputs/learned_50x25/400weight.pth',
            'BR_settings': {'fineness': (50, 25)}
        }
    )

    # -------------------------------------------------------------------------
    # 学習
    # -------------------------------------------------------------------------
    # モデルの保存先
    mpath = args.outputs_path if args.outputs_path[-1:] == '/' \
        else args.outputs_path + '/'

    # logの保存先 2020/01/01 15:30 -> log/20200101_1530に保存
    now = datetime.datetime.now()
    print('log -> {0:%Y%m%d}_{0:%H%M}'.format(now))
    log_dir = args.logdir if args.logdir[-1:] == '/' else args.logdir + '/'
    writer = SummaryWriter(
        log_dir=log_dir + '{0:%Y%m%d}_{0:%H%M}'.format(now)
    )

    # 途中まで学習をしていたらここで読み込み
    if args.start_epoch > 1:
        model.loadmodel('{0:0=3}weight.pth'.format(args.start_epoch), mpath)
        args.start_epoch += 1
    else:
        # 学習前
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

    # 指定epoch数学習
    for epoch in range(args.start_epoch, epochs + 1):
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

        # 5epochごとにモデルを保存
        if (epoch) % 5 == 0:
            model.savemodel('{0:0=3}weight.pth'.format(epoch), mpath)

    writer.close()
    print('finish.')
