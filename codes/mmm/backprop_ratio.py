import numpy as np
import torch
from .datahandler import DataHandler as DH
from tqdm import tqdm


def make_backprop_ratio(train_dataset, category_length, mask,
                        saved=False, save_path='./'):
    '''
    GCNのトレーニングの際，正例が負例に対し極端に少なくなることに対し，
    誤差伝播の重みを変えることで対応するための割合の取得．
    '''
    print('calculating backprop weight ...')

    # データの読み込み
    loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=4
    )

    def _fixed_mask(labels, fmask):
        '''
        誤差を伝播させない部分を指定するマスクの生成
        '''
        labels = labels.data.cpu().numpy()
        labels_y, labels_x = np.where(labels == 1)
        labels_y = np.append(labels_y, labels_y[-1] + 1)
        labels_x = np.append(labels_x, 0)

        fixmask = np.zeros((labels.shape[0] + 1, labels.shape[1]), int)
        row_p, columns_p = labels_y[0], [labels_x[0]]
        fixmask[row_p] = fmask[labels_x[0]]

        for row, column in zip(labels_y[1:], labels_x[1:]):
            if row == row_p:
                columns_p.append(column)
            else:
                if len(columns_p) > 1:
                    for x in columns_p:
                        fixmask[row_p][x] = 0

                row_p, columns_p = row, [column]

            fixmask[row] = fixmask[row] | fmask[column]

        fixmask = fixmask[:-1]

        return fixmask

    # ---入力画像のタグから振り分け------------------------------------------
    counts = np.zeros((category_length, 2))
    for _, label, _ in tqdm(loader):
        if label.sum() == 0:
            counts[:, 0] += 1
            continue

        fix_mask = _fixed_mask(label, mask)
        for idx, flg in enumerate(fix_mask[0]):
            if flg == 1:
                continue

            if label[0][idx] == 0:
                counts[idx][0] += 1
            else:
                counts[idx][1] += 1

    if saved:
        DH.saveNpy(np.array(counts), 'backprop_weight', save_path)

    return np.array(counts, dtype=np.float64)
