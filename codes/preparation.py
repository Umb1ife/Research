from mmm import PrepareVis


def make_backprop_ratio(threshold=0.4, saved=True):
    '''
    GCNのトレーニングの際，正例が負例に対し極端に少なくなることに対し，
    誤差伝播の重みを変えることで対応するための割合の取得．
    '''
    import json
    import numpy as np
    import os
    from mmm import DataHandler as DH
    from tqdm import tqdm

    # ---準備-------------------------------------------------------------------
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_top = '../datas/gcn/'
    outputs_path = path_top + 'inputs/backprop_weight'

    # データの読み込み
    imlist = DH.loadPickle(
        'imlist', directory=path_top + 'outputs/check/'
    )
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    category = [key for key, _ in category.items()]

    # maskの読み込み
    mask = DH.loadPickle(
        '{0:0=2}'.format(int(threshold * 10)),
        path_top + 'inputs/comb_mask'
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

    # ---入力画像のタグから振り分け-----------------------------------------------
    counts = [[0, 0] for _ in range(len(category))]
    for _, label, _ in tqdm(imlist):
        fix_mask = _fixed_mask(label, mask)
        for idx, flg in enumerate(fix_mask[0]):
            if flg == 1:
                continue

            if label[0][idx] == 0:
                counts[idx][0] += 1
            else:
                counts[idx][1] += 1

    if saved:
        DH.saveNpy(
            np.array(counts),
            '{0:0=2}'.format(int(threshold * 10)),
            outputs_path
        )

    return np.array(counts)


if __name__ == "__main__":
    # make_backprop_ratio(0.4)
    # raise Exception
    # -------------------------------------------------------------------------
    # fine_tuning 学習前の準備
    path_top = '../datas/prepare/'
    dp_dict = {
        'local_df': path_top + 'local_df_area16_wocoth.pickle',
        'rep_df': path_top + 'rep_df_area16_wocoth.pickle',
        'all_sim': path_top + 'all_sim.pickle',
        'rep_photo': path_top + 'rep_photo_num.pickle',
        'photo_directory': {
            'rep_train': path_top + 'rep_train',
            'rep_validate': path_top + 'rep_validate',
            'local_train': path_top + 'local_train',
            'local_validate': path_top + 'local_validate'
        }
    }

    PrepareVis().before_finetuning(
        dp_dict, outpath='../datas/',
        # limited_category=['eagle', 'boat']
        limited_category=None
    )
    # -------------------------------------------------------------------------
