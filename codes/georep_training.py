import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from mmm import DataHandler as DH
from mmm import DatasetGeotag
from mmm import RepGeoClassifier
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Fine-tuning')
parser.add_argument('--epochs', '-E', default=200, type=int, metavar='N')
parser.add_argument('--batch_size', '-B', default=1, type=int, metavar='N')
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
    '--sim_threshold', '-Sth', default=0.4, type=float, metavar='N'
)
parser.add_argument(
    '--learning_rate', '-lr', default=0.1, type=float, metavar='N'
)
parser.add_argument('--start_epoch', default=1, type=int, metavar='N')
parser.add_argument(
    '--stage', '-S', default='finetune', type=str, metavar='finetune or gcn'
)


if __name__ == "__main__":
    args = parser.parse_args()

    # パラメータや使用するGPUあたりの設定
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    epochs = args.epochs
    batchsize = args.batch_size * len(args.device_ids.split(','))
    print('Number of GPUs: {}'.format(len(args.device_ids.split(','))))
    numwork = args.workers
    stage = args.stage

    # データの読み込み先
    input_path = args.inputs_path if args.inputs_path[-1:] == '/' \
        else args.inputs_path + '/'
    category = DH.loadJson('category.json', input_path)
    num_class = len(category)
    kwargs_DF = {
        'train': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data_path': input_path
            + 'locate_dataset/{0}_train.pickle'.format(stage)
        },
        'validate': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data_path': input_path
            + 'locate_dataset/{0}_validate.pickle'.format(stage)
        },
    }

    train_dataset = DatasetGeotag(**kwargs_DF['train'])
    # val_dataset = DatasetGeotag(**kwargs_DF['validate'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batchsize,
        num_workers=numwork
    )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     shuffle=True,
    #     batch_size=batchsize,
    #     num_workers=numwork
    # )

    if torch.cuda.is_available():
        train_loader.pin_memory = True
        # val_loader.pin_memory = True
        cudnn.benchmark = True

    # maskの読み込み
    mask = DH.loadPickle(
        'thr_5.pickle', input_path + 'comb_mask_{0}/'.format(stage)
    )

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy('backprop_weight', input_path)
    # bp_weight = DH.loadNpy('thr5_r', input_path + 'backprop_weight')
    # bp_weight = DH.loadNpy('backprop_weight', input_path + 'backprop_weight')

    # 入力位置情報の正規化のためのパラメータ読み込み
    mean, std = DH.loadNpy('normalize_params', input_path)

    model = RepGeoClassifier(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        optimizer=optim.SGD,
        learningrate=args.learning_rate,
        momentum=0.9,
        fix_mask=mask,
        multigpu=True if len(args.device_ids.split(',')) > 1 else False,
        backprop_weight=bp_weight,
        network_setting={'class_num': num_class, 'mean': mean, 'std': std},
    )

    # -------------------------------------------------------------------------
    # 学習
    # -------------------------------------------------------------------------
    # モデルの保存先
    mpath = args.outputs_path if args.outputs_path[-1:] == '/' \
        else args.outputs_path + '/'

    # 途中まで学習をしていたらここで読み込み
    if args.start_epoch > 1:
        model.loadmodel('{0}weight.pth'.format(args.start_epoch), mpath)

    # logの保存先 2020/01/01 15:30 -> log/20200101_1530に保存
    now = datetime.datetime.now()
    print('log -> {0:%Y%m%d}_{0:%H%M}'.format(now))
    log_dir = args.logdir if args.logdir[-1:] == '/' else args.logdir + '/'
    writer = SummaryWriter(
        log_dir=log_dir + '{0:%Y%m%d}_{0:%H%M}'.format(now)
    )

    # 指定epoch数学習
    for epoch in range(args.start_epoch, epochs + 1):
        train_loss, train_recall, train_precision, _, _, _ \
            = model.train(train_loader)
        # val_loss, val_recall, val_precision, fl, pl, al \
        #     = model.validate(val_loader)

        print(
            'epoch %d, loss: %.4f train_recall: %.4f train_precision: %.4f'
            % (
                epoch, train_loss, train_recall, train_precision
            )
        )
        # print(
        #     'epoch %d, loss: %.4f val_loss: %.4f train_recall: %.4f \
        #         val_recall: %.4f train_precision: %.4f val_precision: %.4f'
        #     % (
        #         epoch, train_loss, val_loss, train_recall,
        #         val_recall, train_precision, val_precision
        #     )
        # )

        writer.add_scalar('loss', train_loss, epoch)
        writer.add_scalar('recall', train_recall, epoch)
        writer.add_scalar('precision', train_precision, epoch)

        # writer.add_scalars(
        #     'loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch
        # )
        # writer.add_scalars(
        #     'recall',
        #     {'train_recall': train_recall, 'val_recall': val_recall},
        #     epoch
        # )
        # writer.add_scalars(
        #     'precision',
        #     {
        #         'train_precision': train_precision,
        #         'val_precision': val_precision
        #     },
        #     epoch
        # )

        # 5epochごとにモデルを保存
        if (epoch) % 5 == 0:
            filename = str(epoch) + 'weight.pth'
            model.savemodel(filename, mpath)

    writer.close()
    print('finish.')
