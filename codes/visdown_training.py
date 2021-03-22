import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from mmm import DataHandler as DH
from mmm import DatasetFlickr
from mmm import MakeBPWeight
from mmm import VisGCN
from mmm import VisUtils as VU
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', '-E', default=20, type=int, metavar='N')
parser.add_argument('--batch_size', '-B', default=64, type=int, metavar='N')
parser.add_argument('--load_mask', action='store_true')
parser.add_argument('--load_backprop_weight', action='store_true')
parser.add_argument(
    '--device_ids', '-D', default='0, 1, 2, 3', type=str, metavar="'i, j, k'"
)
parser.add_argument(
    '--inputs_path', '-I', default='../datas/vis_down/inputs', type=str,
    metavar='path of directory containing input data'
)
parser.add_argument(
    '--outputs_path', '-O', default='../datas/vis_down/outputs/learned',
    type=str, metavar='path of directory trained model saved'
)
parser.add_argument(
    '--logdir', '-L', default='../datas/vis_down/log', type=str,
    metavar='path of directory log saved'
)
parser.add_argument(
    '--learning_rate', '-lr', default=1, type=float, metavar='N'
)
parser.add_argument(
    '--sim_threshold', '-Sth', default=0.4, type=float, metavar='N'
)
parser.add_argument('--start_epoch', default=1, type=int, metavar='N')
parser.add_argument('--workers', '-W', default=4, type=int, metavar='N')


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
    input_path = args.inputs_path if args.inputs_path[-1:] == '/' \
        else args.inputs_path + '/'
    category = DH.loadJson('category.json', input_path)
    rep_category = DH.loadJson('upper_category.json', input_path)

    vis_down_train = VU.down_anno(category, rep_category, 'train')
    vis_down_validate = VU.down_anno(category, rep_category, 'validate')
    DH.saveJson(vis_down_train, 'train_anno', input_path)
    DH.saveJson(vis_down_validate, 'validate_anno', input_path)
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
    mask = DH.loadPickle('04.pickle', input_path) if args.load_mask else None
    mask = VU.down_mask(
        rep_category, category,
        sim_thr=args.sim_threshold,
        saved=True
    ) if mask is None else mask

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy('backprop_weight', input_path) \
        if args.load_backprop_weight else None
    bp_weight = bp_weight if bp_weight is not None \
        else MakeBPWeight(train_dataset, num_class, mask, False, input_path)

    # 学習で用いるデータの設定や読み込み先
    gcn_settings = {
        'category': category,
        'rep_category': rep_category,
        'relationship': DH.loadPickle('relationship.pickle', input_path),
        'rep_weight': torch.load(input_path + 'rep_weight.pth'),
        'feature_dimension': 2048
    }

    # modelの設定
    model = VisGCN(
        class_num=num_class,
        loss_function=MyLossFunction(),
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

    # 途中まで学習をしていたらここで読み込み
    if args.start_epoch > 1:
        model.loadmodel('{0:0=3}weight'.format(args.start_epoch), mpath)
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

    # 学習
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

        model.savemodel('{0:0=3}weight'.format(epoch), mpath)

    writer.close()
    print('finish.')
