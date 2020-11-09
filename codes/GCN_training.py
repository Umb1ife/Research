import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from mmm import DataHandler as DH
from mmm import DatasetFlickr
from mmm import MultiLabelGCN
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', '-B', default=64, type=int, metavar='N')
parser.add_argument(
    '--device_ids', '-D', default='0, 1, 2, 3', type=str, metavar="'i, j, k'"
)
parser.add_argument(
    '--inputs_path', '-I', default='../datas/gcn/inputs', type=str,
    metavar='path of directory containing input data'
)
parser.add_argument(
    '--outputs_path', '-O', default='../datas/gcn/outputs/learned',
    type=str, metavar='path of directory trained model saved'
)
parser.add_argument(
    '--logdir', '-L', default='../datas/gcn/log', type=str,
    metavar='path of directory log saved'
)
parser.add_argument('--epochs', '-E', default=20, type=int, metavar='N')
parser.add_argument(
    '--learning_rate', '-lr', default=0.1, type=float, metavar='N'
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
        '{0:0=2}'.format(int(args.sim_threshold * 10)),
        input_path + 'comb_mask/'
    )

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy(
        '{0:0=2}'.format(int(args.sim_threshold * 10)),
        input_path + 'backprop_weight/'
    )

    # 学習で用いるデータの設定や読み込み先
    gcn_settings = {
        'num_class': num_class,
        'filepaths': {
            'category': input_path + 'category.json',
            'upper_category': input_path + 'upper_category.json',
            'relationship': input_path + 'relationship.pickle',
            # 'learned_weight': input_path + 'learned/200cnn.pth'
            'learned_weight': input_path + '200cnn.pth'
        },
        'feature_dimension': 2048
    }

    # modelの設定
    model = MultiLabelGCN(
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

    model.savemodel('{0:0=3}weight'.format(0), mpath)
    for epoch in range(args.start_epoch, epochs + 1):
        train_loss, train_recall, train_precision, fl, pl, al \
            = model.train(train_loader)
        val_loss, val_recall, val_precision, fl, pl, al \
            = model.validate(val_loader)
        print('epoch: {0}'.format(epoch))
        print('loss: {0}, recall: {1}, precision: {2}'.format(
            train_loss, train_recall, train_precision
        ))
        print('loss: {0}, recall: {1}, precision: {2}'.format(
            val_loss, val_recall, val_precision
        ))
        print('--------------------------------------------------------------')

        writer.add_scalars(
            'loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch
        )
        writer.add_scalars(
            'recall',
            {'train_recall': train_recall, 'val_recall': val_recall},
            epoch
        )
        writer.add_scalars(
            'precision',
            {
                'train_precision': train_precision,
                'val_precision': val_precision
            },
            epoch
        )

        spath = mpath + 'fpa_list/{0:0=2}/{1:0=3}'.format(
            int(args.sim_threshold * 10), epoch
        )
        model.savemodel('{0:0=3}weight'.format(epoch), mpath)
        DH.saveNpy(fl, 'fl', spath)
        DH.saveNpy(pl, 'pl', spath)
        DH.saveNpy(al, 'al', spath)
