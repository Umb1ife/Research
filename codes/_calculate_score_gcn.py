import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
from mmm import DataHandler as DH
from mmm import ImbalancedDataSampler as IDS, DatasetFlickr
from mmm import MultiLabelGCN
from torchvision import transforms


np.set_printoptions(threshold=np.inf)
path_top = '../datas/gcn/'

parser = argparse.ArgumentParser(description='Fine-tuning')
# parser.add_argument('--epochs', '-E', default=200, type=int, metavar='N')
parser.add_argument('--batch_size', '-B', default=64, type=int, metavar='N')
parser.add_argument(
    '--device_ids', '-D', default='0, 1, 2', type=str, metavar="'i, j, k'"
)
parser.add_argument('--workers', '-W', default=4, type=int, metavar='N')
parser.add_argument(
    '--learning_rate', '-lr', default=0.01, type=float, metavar='N'
)
parser.add_argument('--start_epoch', '-S', default=20, type=int, metavar='N')
parser.add_argument('--threshold', '-th', default=0.1, type=float, metavar='N')


def cal_precision(threshold, now_epoch=1):
    fpa_path = path_top + 'outputs2/fpa_list/{0:0=2}/{1:0=3}/'.format(
        int(threshold * 10), now_epoch
    )
    _, pl, al = DH.loadNpy('fl', fpa_path), \
        DH.loadNpy('pl', fpa_path), DH.loadNpy('al', fpa_path)
    class_num = len(al[0][0])

    # upper_category = json.load(
    #     open(path_top + 'inputs/upper_category.json', 'r')
    # )
    # category = json.load(open(path_top + 'inputs/category.json', 'r'))

    ans_predict_table = np.zeros((class_num, class_num), dtype=int)
    pl = [item for batch in pl for item in batch]
    al = [item for batch in al for item in batch]

    for p, a in zip(pl, al):
        p = np.where(p == 1)[0]
        a = np.where(a == 1)[0]
        for ans in a:
            for predict in p:
                ans_predict_table[ans][predict] += 1

    # DH.saveNpy(ans_predict_table, 'ap_table', path_top + 'outputs2/check/')
    precision_mask = make_mask(threshold)
    remove_sim_matrix = ans_predict_table * precision_mask
    diagonal_matrix = ans_predict_table * np.identity(class_num, dtype=int)
    # print(np.sum(remove_sim_matrix, axis=0))
    # print(np.sum(diagonal_matrix, axis=0))
    # cp = np.sum(diagonal_matrix, axis=0) / np.sum(remove_sim_matrix, axis=0)
    print(np.sum(diagonal_matrix) / np.sum(remove_sim_matrix))

    # ucp = []
    # for label, _ in upper_category.items():
    #     ucp.append(cp[category[label]])

    # print(ucp)

    # import matplotlib.pyplot as plt
    # plt.hist(
    #     np.sum(diagonal_matrix, axis=0) / np.sum(remove_sim_matrix, axis=0)
    # )
    # plt.show()

    # return np.sum(diagonal_matrix, axis=0), np.sum(remove_sim_matrix, axis=0)
    return ans_predict_table, diagonal_matrix, remove_sim_matrix


def make_mask(threshold):
    # upper_category = json.load(
    #     open(path_top + 'inputs/upper_category.json', 'r')
    # )
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    all_sim = DH.loadPickle(path_top + 'inputs/all_sim_revised.pickle')
    cat = list(category.keys())
    class_num = len(cat)
    mask = np.ones((class_num, class_num), dtype=int)

    templist = all_sim.values.tolist()
    left = [item[0][0] for item in templist]
    right = [item[0][1] for item in templist]
    sim = [item[1] for item in templist]

    df = pd.DataFrame(zip(left, right, sim), columns=['left', 'right', 'sim'])
    df = df[(df['left'].isin(cat)) & (df['right'].isin(cat))]
    df = df[df.sim >= threshold]

    for key, value in category.items():
        simtags = df[(df['left'] == key) | (df['right'] == key)]
        simtags = list(set(
            simtags['left'].tolist() + simtags['right'].tolist()
        ))
        simtags = [item for item in simtags if item != key]

        for simkey in simtags:
            mask[category[simkey]][value] = 0

    return mask


def get_fpa(saved=False, now_epoch=1, threshold=0.1):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    batchsize = args.batch_size * len(args.device_ids.split(','))
    print('Number of GPUs: {}'.format(len(args.device_ids.split(','))))
    numwork = args.workers

    # データの読み込み先
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    kwargs_DF = {
        'train': {
            'filenames': {
                'Annotation': path_top + 'inputs/train_anno.json',
                'Category_to_Index': path_top + 'inputs/category.json'
            },
            'phase': 'train',
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
            'image_path': path_top + 'inputs/images/train/'
        },
        'validate': {
            'filenames': {
                'Annotation': path_top + 'inputs/validate_anno.json',
                'Category_to_Index': path_top + 'inputs/category.json'
            },
            'phase': 'validate',
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
            'image_path': path_top + 'inputs/images/validate/'
        }
    }

    train_dataset = DatasetFlickr(**kwargs_DF['train'])
    val_dataset = DatasetFlickr(**kwargs_DF['validate'])
    num_class = val_dataset.num_category()

    # not all_rep
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batchsize, shuffle=False, num_workers=numwork
    )
    # all_rep
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=IDS(train_dataset, callback_get_label=lambda d, i: None),
        batch_size=batchsize,
        num_workers=numwork
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=IDS(val_dataset, callback_get_label=lambda d, i: None),
        batch_size=batchsize,
        num_workers=numwork
    )

    if torch.cuda.is_available():
        train_loader.pin_memory = True
        val_loader.pin_memory = True
        cudnn.benchmark = True

    # maskの読み込み
    with open(path_top + 'inputs/comb_mask/{0:0=2}.pickle'.format(
              int(threshold * 10)), 'rb') as f:
        mask = pickle.load(f)

    # 学習で用いるデータの設定や読み込み先
    gcn_settings = {
        'class_num': num_class,
        'filepaths': {
            'category': path_top + 'inputs/category.json',
            'upper_category': path_top + 'inputs/upper_category.json',
            'relationship': path_top + 'inputs/relationship.pickle',
            'learned_weight': path_top + 'inputs/learned/200cnn.pth'
        },
        'feature_dimension': 2048
    }

    # # modelの設定
    # model = MultiLabelGCN(
    #     class_num=num_class,
    #     loss_function=MyLossFunction(),
    #     optimizer=optim.SGD,
    #     learningrate=args.learning_rate,
    #     fix_mask=mask,
    #     multigpu=True if len(args.device_ids.split(',')) > 1 else False
    # )

    # modelの設定
    model = MultiLabelGCN(
        # batch_size=batchsize,
        class_num=num_class,
        loss_function=MyLossFunction(),
        optimizer=optim.SGD,
        learningrate=args.learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        fix_mask=mask,
        gcn_settings=gcn_settings,
        multigpu=True if len(args.device_ids.split(',')) > 1 else False
    )

    # モデルの保存先
    mpath = path_top + 'outputs2/learned/'

    # 途中まで学習をしていたらここで読み込み
    if now_epoch > 0:
        model.loadmodel('{0:0=3}weight.pickle'.format(now_epoch), mpath)

    # val_loss, val_recall, val_precision, fl, pl, al \
    #     = model.validate(val_loader)
    val_loss, val_recall, val_precision, fl, pl, al \
        = model.validate(train_loader)
    if saved:
        spath = path_top + 'outputs2/fpa_list/{0:0=2}/{1:0=3}'.format(
            int(args.threshold * 10), now_epoch
        )
        DH.saveNpy(fl, 'fl', spath)
        DH.saveNpy(pl, 'pl', spath)
        DH.saveNpy(al, 'al', spath)

    print(
        'epoch %d, val_loss: %.4f val_recall: %.4f val_precision: %.4f'
        % (now_epoch, val_loss, val_recall, val_precision)
    )


if __name__ == "__main__":
    # データが存在しなければ生成
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    path_top = '../datas/gcn/'
    # if not os.path.isfile(
    #     path_top + 'outputs2/fpa_list/{0:0=2}/{1:0=3}/fl.npy'.format(
    #         int(args.threshold * 10), now_epoch
    #     )
    # ):
    #     get_fpa()
    #     print('generated.')
    # else:
    #     print('exists.')

    # -------------------------------------------------------------------------
    # get_fpa(True, 0, 0.1)

    outdir = path_top + 'outputs2/check/pr_matrix'

    # m_0100 = cal_precision(0.1, 0)
    # m_0101 = cal_precision(0.1, 1)
    # m_0120 = cal_precision(0.1, 20)
    # DH.saveNpy(m_0100, '0100', outdir)
    # DH.saveNpy(m_0101, '0101', outdir)
    # DH.saveNpy(m_0120, '0120', outdir)

    # m_0301 = cal_precision(0.3, 1)
    # m_0320 = cal_precision(0.3, 20)
    # DH.saveNpy(m_0301, '0301', outdir)
    # DH.saveNpy(m_0320, '0320', outdir)

    m_0420 = cal_precision(0.4, 20)
    DH.saveNpy(m_0420, '0420', outdir)
    # -------------------------------------------------------------------------
    print('finish.')
