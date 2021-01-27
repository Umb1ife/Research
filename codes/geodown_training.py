import argparse
import datetime
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from mmm import DataHandler as DH
from mmm import DatasetGeotag
from mmm import GeotagGCN
from mmm import GeoUtils as GU
from mmm import MakeBPWeight
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', '-E', default=20, type=int, metavar='N')
parser.add_argument('--batch_size', '-B', default=1, type=int, metavar='N')
parser.add_argument('--load_mask', action='store_true')
parser.add_argument('--load_backprop_weight', action='store_true')
parser.add_argument(
    '--device_ids', '-D', default='0', type=str, metavar="'i, j, k'"
)
parser.add_argument(
    '--inputs_path', '-I', default='../datas/geo_down/inputs', type=str,
    metavar='path of directory containing input data'
)
parser.add_argument(
    '--outputs_path', '-O', default='../datas/geo_down/outputs/learned',
    type=str, metavar='path of directory trained model saved'
)
parser.add_argument(
    '--logdir', '-L', default='../datas/geo_down/log', type=str,
    metavar='path of directory log saved'
)
parser.add_argument(
    '--learning_rate', '-lr', default=0.1, type=float, metavar='N'
)
parser.add_argument('--start_epoch', default=1, type=int, metavar='N')
parser.add_argument('--workers', '-W', default=4, type=int, metavar='N')


def limited_category(reps, lda='../datas/bases/local_df_area16_wocoth_new'):
    gr = DH.loadPickle('geo_relationship.pickle', '../datas/bases')
    vis_local = DH.loadJson('category', '../datas/gcn/inputs')
    vis_rep = DH.loadJson('upper_category', '../datas/gcn/inputs')
    local = set(vis_local) - set(vis_rep)

    down = []
    for item in reps:
        down.extend(gr[item])

    down = set(down) & set(local)

    lda = DH.loadPickle(lda)
    down = [item for item in down if len(lda['geo'][item]) > 0]
    down = sorted(list(set(down) | set(reps)))

    return {key: idx for idx, key in enumerate(down)}


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 初期設定
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # パラメータや使用するGPUあたりの設定
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    epochs = args.epochs
    batchsize = args.batch_size * len(args.device_ids.split(','))
    print('Number of GPUs: {}'.format(len(args.device_ids.split(','))))
    numwork = args.workers

    # データの読み込み先
    base_path = '../datas/bases/'
    input_path = args.inputs_path if args.inputs_path[-1:] == '/' \
        else args.inputs_path + '/'

    rep_category = DH.loadJson('upper_category.json', input_path)
    category = DH.loadJson('category.json', input_path)
    # rep_category = {'lasvegas': 0, 'newyorkcity': 1, 'seattle': 2}
    # category = limited_category(rep_category)
    category = limited_category(
        rep_category,
        lda='../datas/geo_down/inputs/local_df_area16_wocoth_new'
    )
    num_class = len(category)

    # データの作成
    geo_down_train = GU.down_dataset(
        rep_category, category, 'train',
        # base_path=base_path
        base_path=input_path
    )
    geo_down_validate = GU.down_dataset(
        rep_category, category, 'validate',
        # base_path=base_path
        base_path=input_path
    )
    DH.savePickle(geo_down_train, 'geo_down_train', input_path)
    DH.savePickle(geo_down_validate, 'geo_down_validate', input_path)

    kwargs_DF = {
        'train': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data': geo_down_train
        },
        'validate': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data': geo_down_validate
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
    mask = GU.down_mask(rep_category, category) if mask is None else mask

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy('backprop_weight', input_path) \
        if args.load_backprop_weight else None
    bp_weight = bp_weight if bp_weight is not None \
        else MakeBPWeight(train_dataset, num_class, mask, True, input_path)
    bp_weight = np.power(bp_weight, 2)

    # 入力位置情報の正規化のためのパラメータ読み込み
    mean, std = DH.loadNpy('normalize_params', input_path)

    # 学習で用いるデータの設定や読み込み先
    gcn_settings = {
        'category': category,
        'rep_category': rep_category,
        'filepaths': {
            'relationship': base_path + 'geo_relationship.pickle',
            # 'learned_weight': input_path + '020weight.pth'
            'learned_weight': input_path + '200weight.pth'
        },
        'feature_dimension': 30,
        'simplegeonet_settings': {
            'class_num': len(rep_category), 'mean': mean, 'std': std
        }
    }

    # modelの設定
    model = GeotagGCN(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        optimizer=optim.SGD,
        learningrate=args.learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        fix_mask=mask,
        network_setting=gcn_settings,
        multigpu=True if len(args.device_ids.split(',')) > 1 else False,
        backprop_weight=bp_weight
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

    # 学習前
    model.savemodel('000weight.pth', mpath)
    train_loss, train_recall, train_precision, _, _, _ \
        = model.validate(train_loader)
    val_loss, val_recall, val_precision, _, _, _ \
        = model.validate(val_loader)
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

    # 学習
    for epoch in range(args.start_epoch, epochs + 1):
        train_loss, train_recall, train_precision, _, _, _ \
            = model.train(train_loader)
        val_loss, val_recall, val_precision, _, _, _ \
            = model.validate(val_loader)
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

        model.savemodel('{0:0=3}weight'.format(epoch), mpath)

    writer.close()
    print('finish.')
