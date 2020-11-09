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
from mmm import FinetuneModel
from torchvision import transforms


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
parser.add_argument('--start_epoch', '-S', default=1, type=int, metavar='N')
parser.add_argument('--threshold', '-th', default=0.4, type=float, metavar='N')


def cal_precision(threshold):
    fpa_path = path_top + 'outputs/fpa_list/{:0=3}/'.format(now_epoch)
    _, pl, al = DH.loadNpy('fl', fpa_path), \
        DH.loadNpy('pl', fpa_path), DH.loadNpy('al', fpa_path)
    class_num = len(al[0][0])

    predict_ans_table = np.zeros((class_num, class_num), dtype=int)
    pl = [item for batch in pl for item in batch]
    al = [item for batch in al for item in batch]
    # print(len(al), len(pl))

    for p, a in zip(pl, al):
        p = np.where(p == 1)[0]
        a = np.where(a == 1)[0]
        for ans in a:
            for predict in p:
                predict_ans_table[ans][predict] += 1

    precision_mask = make_precision_mask(threshold)
    # print(precision_mask)
    # print(np.where(precision_mask == 0))
    # raise Exception
    remove_sim_matrix = predict_ans_table * precision_mask
    diagonal_matrix = predict_ans_table * np.identity(class_num, dtype=int)
    print(np.sum(remove_sim_matrix, axis=0))
    print(np.sum(diagonal_matrix, axis=0))
    print(np.sum(diagonal_matrix, axis=0) / np.sum(remove_sim_matrix, axis=0))
    print(np.sum(diagonal_matrix) / np.sum(remove_sim_matrix))
    print(np.average(
        np.sum(diagonal_matrix, axis=0) / np.sum(remove_sim_matrix, axis=0)
    ))

    # import matplotlib.pyplot as plt
    # plt.hist(
    #     np.sum(diagonal_matrix, axis=0) / np.sum(remove_sim_matrix, axis=0)
    # )
    # plt.show()


def make_precision_mask(threshold):
    file_path = path_top + 'inputs/all_sim_revised.pickle'
    json_path = path_top + 'inputs/category.json'
    all_sim = DH.loadPickle(file_path)
    category = json.load(open(json_path, 'r'))
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


def get_fpa(saved=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    batchsize = args.batch_size * len(args.device_ids.split(','))
    print('Number of GPUs: {}'.format(len(args.device_ids.split(','))))
    numwork = args.workers

    # データの読み込み
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
            # 'image_path': path_top + 'inputs/images/'
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
            # 'image_path': path_top + 'inputs/images/'
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
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     sampler=IDS(train_dataset, callback_get_label=lambda d, i: None),
    #     batch_size=batchsize,
    #     num_workers=numwork
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     sampler=IDS(val_dataset, callback_get_label=lambda d, i: None),
    #     batch_size=batchsize,
    #     num_workers=numwork
    # )

    if torch.cuda.is_available():
        train_loader.pin_memory = True
        val_loader.pin_memory = True
        cudnn.benchmark = True

    # maskの読み込み
    with open(path_top + 'inputs/comb_mask/{:0=2}.pickle'.format(
              int(args.threshold * 10)), 'rb') as f:
        mask = pickle.load(f)

    # modelの設定
    model = FinetuneModel(
        class_num=num_class,
        loss_function=MyLossFunction(),
        optimizer=optim.SGD,
        learningrate=args.learning_rate,
        fix_mask=mask,
        multigpu=True if len(args.device_ids.split(',')) > 1 else False
    )

    # モデルの保存先
    mpath = path_top + 'outputs/learned/'

    # 途中まで学習をしていたらここで読み込み
    if now_epoch > 0:
        model.loadmodel('{}cnn.pth'.format(now_epoch), mpath)

    val_loss, val_recall, val_precision, fl, pl, al \
        = model.validate(val_loader)
    # val_loss, val_recall, val_precision, fl, pl, al \
    #     = model.validate(train_loader)
    spath = path_top + 'outputs/fpa_list/{:0=3}'.format(now_epoch)
    if saved:
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
    now_epoch = args.start_epoch
    path_top = '../datas/fine_tuning/'
    if not os.path.isfile(
        path_top + 'outputs/fpa_list/{:0=3}/fl.npy'.format(now_epoch)
    ):
        get_fpa(saved=True)
        print('generated.')
    else:
        print('exists.')

    # -------------------------------------------------------------------------
    get_fpa(saved=True)
    cal_precision(args.threshold)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    print('finish.')
