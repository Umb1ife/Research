import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from mmm import DataHandler as DH
from mmm import DatasetGeobase
from mmm import GeoBaseNet
from mmm import GeoUtils as GU
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='pretrained model')
parser.add_argument('--epochs', '-E', default=200, type=int, metavar='N')
parser.add_argument('--batch_size', '-B', default=256, type=int, metavar='N')
parser.add_argument('--load_mask', action='store_true')
parser.add_argument('--load_backprop_weight', action='store_true')
parser.add_argument(
    '--device_ids', '-D', default='0, 1, 2, 3', type=str, metavar="'i, j, k'"
)
parser.add_argument(
    '--inputs_path', '-I', default='../datas/geo_base/inputs', type=str,
    metavar='path of directory containing input data'
)
parser.add_argument(
    '--outputs_path', '-O', default='../datas/geo_base/outputs/learned',
    type=str, metavar='path of directory trained model saved'
)
parser.add_argument(
    '--logdir', '-L', default='../datas/geo_base/log', type=str,
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
    base_setting = {
        'x_range': (-175, -64),
        'y_range': (18, 71),
        'fineness': (20, 20),
        'numdata_sqrt_oneclass': 32
    }
    base_train, (mean, std) = GU.base_dataset(**base_setting)
    num_class = base_setting['fineness'][0] * base_setting['fineness'][1]
    train_dataset = DatasetGeobase(
        class_num=num_class,
        transform=torch.tensor,
        data=base_train,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batchsize,
        num_workers=numwork
    )

    if torch.cuda.is_available():
        train_loader.pin_memory = True
        cudnn.benchmark = True

    BR_settings = {
        'x_range': (-175, -64),
        'y_range': (18, 71),
        'fineness': (20, 20),
        'mean': mean,
        'std': std
    }

    model = GeoBaseNet(
        class_num=num_class,
        loss_function=torch.nn.NLLLoss(),
        optimizer=optim.SGD,
        learningrate=args.learning_rate,
        momentum=0.9,
        multigpu=True if len(args.device_ids.split(',')) > 1 else False,
        network_setting=BR_settings,
    )

    # -------------------------------------------------------------------------
    # 学習
    # -------------------------------------------------------------------------
    # モデルの保存先
    mpath = args.outputs_path if args.outputs_path[-1:] == '/' \
        else args.outputs_path + '/'

    # 途中まで学習をしていたらここで読み込み
    if args.start_epoch > 1:
        model.loadmodel('{0:0=3}weight.pth'.format(args.start_epoch), mpath)

    # logの保存先 2020/01/01 15:30 -> log/20200101_1530に保存
    now = datetime.datetime.now()
    print('log -> {0:%Y%m%d}_{0:%H%M}'.format(now))
    log_dir = args.logdir if args.logdir[-1:] == '/' else args.logdir + '/'
    writer = SummaryWriter(
        log_dir=log_dir + '{0:%Y%m%d}_{0:%H%M}'.format(now)
    )

    # 指定epoch数学習
    model.savemodel('000weight.pth', mpath)
    train_loss, train_recall, train_precision = model.validate(train_loader)
    print('epoch: {0}'.format(0))
    print('loss: {0}, recall: {1}, precision: {2}'.format(
        train_loss, train_recall, train_precision
    ))

    writer.add_scalar('loss', train_loss, 0)
    writer.add_scalar('recall', train_recall, 0)
    writer.add_scalar('precision', train_precision, 0)
    print('------------------------------------------------------------------')

    # 学習
    for epoch in range(args.start_epoch, epochs + 1):
        train_loss, train_recall, train_precision = model.train(train_loader)

        print('epoch: {0}'.format(epoch))
        print('loss: {0}, recall: {1}, precision: {2}'.format(
            train_loss, train_recall, train_precision
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
