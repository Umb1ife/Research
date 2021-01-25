def make_imlist(phase='train', saved=True):
    import os
    import torch
    from mmm import DataHandler as DH
    from mmm import DatasetFlickr
    from torchvision import transforms
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_top = '../datas/gcn/'

    # データの読み込み先
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    kwargs_DF = {
        'train': {
            'filenames': {
                'Annotation': path_top + 'inputs/train_anno.json',
                'Category_to_Index': path_top + 'inputs/category.json'
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
            'image_path': path_top + 'inputs/images/train/'
        },
        'validate': {
            'filenames': {
                'Annotation': path_top + 'inputs/validate_anno.json',
                'Category_to_Index': path_top + 'inputs/category.json'
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
            'image_path': path_top + 'inputs/images/validate/'
        }
    }

    dataset = DatasetFlickr(**kwargs_DF[phase])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    imlist = [
        (image, label, filename) for image, label, filename in tqdm(loader)
    ]

    if saved:
        DH.savePickle(imlist, 'imlist', directory=path_top + 'outputs/check/')

    return imlist


def get_predicts(phase='train', threshold=0.4, check_epoch=1,
                 ranges=(0, 12), learning_rate=0.1):
    '''
    画像をmodelに入力したときの画像表示，正解ラベル，正解ラベルの順位・尤度，全ラベルの尤度を出力

    Arguments:
        phase {'train' or 'validate'} -- 入力が学習用画像か評価用画像かの指定
        threshold {0.1 - 0.9} -- maskのしきい値．予め作成しておくこと
        check_epoch {1 - 20} -- どのepochの段階で評価するかの指定
        ranges {(int, int)} -- 画像集合の何番目から何番目を取ってくるかの指定
    '''
    # ---準備-------------------------------------------------------------------
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    # import torch
    import torch.optim as optim
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import DatasetFlickr
    from mmm import MultiLabelGCN
    from PIL import Image
    from torchvision import transforms

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_top = '../datas/gcn/'
    ranges = ranges if ranges[1] > ranges[0] else (ranges[1], ranges[0])
    ranges = ranges if ranges[1] != ranges[0] else (ranges[0], ranges[0] + 1)

    # データの読み込み先
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    kwargs_DF = {
        'train': {
            'filenames': {
                'Annotation': path_top + 'inputs/train_anno.json',
                'Category_to_Index': path_top + 'inputs/category.json'
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
            'image_path': path_top + 'inputs/images/train/'
        },
        'validate': {
            'filenames': {
                'Annotation': path_top + 'inputs/validate_anno.json',
                'Category_to_Index': path_top + 'inputs/category.json'
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
            'image_path': path_top + 'inputs/images/validate/'
        }
    }

    dataset = DatasetFlickr(**kwargs_DF[phase])
    num_class = dataset.num_category()
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # maskの読み込み
    mask = DH.loadPickle(
        '{0:0=2}'.format(int(threshold * 10)),
        path_top + 'inputs/comb_mask/'
    )

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy(
        '{0:0=2}'.format(int(threshold * 10)),
        path_top + 'inputs/backprop_weight/'
    )

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

    # modelの設定
    model = MultiLabelGCN(
        class_num=num_class,
        loss_function=MyLossFunction(),
        optimizer=optim.SGD,
        learningrate=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        fix_mask=mask,
        network_setting=gcn_settings,
        multigpu=False,
        backprop_weight=bp_weight
    )

    # モデルの保存先
    mpath = path_top + 'outputs/learned/th{0:0=2}/'.format(int(threshold * 10))

    # 途中まで学習をしていたらここで読み込み
    if check_epoch > 0:
        model.loadmodel('{0:0=3}weight'.format(check_epoch), mpath)

    def get_row_col(length):
        '''
        画像を並べて表示する際の行列数の計算
        '''
        sq = int(np.sqrt(length))
        if sq ** 2 == length:
            return sq, sq

        if (sq + 1) * sq >= length:
            return sq, sq + 1
        else:
            return sq + 1, sq + 1

    # ---平均ラベル数の計算------------------------------------------------------
    # ave, cnt = 0, 0
    # mx, mi = 0, np.inf
    # scores = []
    # for i, (image, label, filename) in enumerate(loader):
    #     score = sum(model.predict(image, labeling=True)[0])
    #     ave += score
    #     cnt += 1
    #     mx = score if score > mx else mx
    #     mi = score if score < mi else mi
    #     scores.append(score)

    # plt.hist(scores)
    # plt.show()
    # print('average: {0}'.format(ave / cnt))
    # print('max, min: {0}, {1}'.format(mx, mi))

    # ---画像・正解ラベル・予測ラベルの表示---------------------------------------
    # imlist = [(image, label, filename) for image, label, filename in loader]
    # DH.savePickle(imlist, 'imlist', directory=path_top + 'outputs/check/')
    # imlist = DH.loadPickle('imlist', directory='../datas/geo_rep/inputs')
    imlist = DH.loadPickle('imlist', directory=path_top + 'outputs/check/')
    imlist = imlist[ranges[0]:ranges[1]]
    category = json.load(
        open(kwargs_DF[phase]['filenames']['Category_to_Index'], 'r')
    )
    category = [key for key, _ in category.items()]

    results = []
    row, col = get_row_col(len(imlist))
    for idx, (image, label, filename) in enumerate(imlist):
        # 画像の表示
        print(kwargs_DF[phase]['image_path'], filename[0])
        image_s = Image.open(
            os.path.join(kwargs_DF[phase]['image_path'], filename[0])
        ).convert('RGB')
        plt.subplot(row, col, idx + 1)
        plt.imshow(image_s)
        plt.axis('off')

        # 正解ラベル
        label = [lname for flg, lname in zip(label[0], category) if flg == 1]

        # 予測ラベル
        pred = model.predict(image)[0]
        pred = [
            (likelihood, lname) for likelihood, lname in zip(pred, category)
            if likelihood != 0
        ]
        pred = sorted(pred, reverse=True)
        # 予測ラベルのうちtop nまでのもの
        toppred = [(item[1], item[0]) for item in pred]

        # 正解ラベルが予測ではどの順位にあるか
        prank = {tag: (idx, llh) for idx, (llh, tag) in enumerate(pred)}
        prank = [prank[lbl] for lbl in label]

        result = {
            'filename': filename[0],
            'image': image_s,
            'tag': label,
            'tags_rank': prank,
            'predict': toppred
        }

        results.append(result)

    return results


def get_training_images(threshold=0.1, phase='train'):
    '''
    あるクラスについてのトレーニングデータを確認．
    thresholdによって出力サイズがバカでかくなることもあるため注意．

    Arguments:
        phase {'train' or 'validate'} -- 入力が学習用画像か評価用画像かの指定
        threshold {0.1 - 0.9} -- maskのしきい値．maskは予め作成しておくこと
    '''
    # ---準備-------------------------------------------------------------------
    import json
    import numpy as np
    import os
    import pickle
    import shutil
    from mmm import DataHandler as DH
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_top = '../datas/gcn/'

    # データの読み込み
    imlist = DH.loadPickle('imlist', directory=path_top + 'outputs/check/')
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    category = [key for key, _ in category.items()]

    # maskの読み込み
    with open(path_top + 'inputs/comb_mask/{0:0=2}.pickle'.format(
              int(threshold * 10)), 'rb') as f:
        mask = pickle.load(f)

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
    outputs_path = path_top \
        + 'outputs/check/images/th{0:0=2}/'.format(int(threshold * 10))

    for _, label, filename in tqdm(imlist):
        fix_mask = _fixed_mask(label, mask)
        for idx, flg in enumerate(fix_mask[0]):
            if flg == 1:
                continue

            path_fin = category[idx] + '/zero' if label[0][idx] == 0 \
                else category[idx] + '/one'
            os.makedirs(outputs_path + path_fin, exist_ok=True)
            shutil.copy(
                path_top + 'inputs/images/' + phase + '/' + filename[0],
                outputs_path + path_fin
            )


def class_precision(threshold=0.4, epoch=20):
    '''
    maskによりunknownと指定された画像を除いた場合での，
    クラス毎の予測ラベルと正解を比較
    '''
    # -------------------------------------------------------------------------
    # 準備
    import json
    import numpy as np
    import os
    import pickle
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import MultiLabelGCN
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_top = '../datas/gcn/'

    # maskの読み込み
    with open(path_top + 'inputs/comb_mask/{0:0=2}.pickle'.format(
              int(threshold * 10)), 'rb') as f:
        mask = pickle.load(f)

    # 誤差伝播の重みの読み込み
    with open(path_top + 'inputs/backprop_weight/{0:0=2}.npy'.format(
              int(threshold * 10)), 'rb') as f:
        bp_weight = np.load(f)

    # modelの設定
    gcn_settings = {
        'class_num': 3100,
        'filepaths': {
            'category': path_top + 'inputs/category.json',
            'upper_category': path_top + 'inputs/upper_category.json',
            'relationship': path_top + 'inputs/relationship.pickle',
            'learned_weight': path_top + 'inputs/learned/200cnn.pth'
        },
        'feature_dimension': 2048
    }
    model = MultiLabelGCN(
        class_num=3100,
        loss_function=MyLossFunction(),
        learningrate=0.1,
        weight_decay=1e-4,
        fix_mask=mask,
        network_setting=gcn_settings,
        multigpu=False,
        backprop_weight=bp_weight
    )
    if epoch > 0:
        model.loadmodel(
            '{0:0=3}weight'.format(epoch),
            path_top + 'outputs/learned/th{0:0=2}'.format(int(threshold * 10))
        )

    # データの読み込み
    imlist = DH.loadPickle('imlist', directory=path_top + 'outputs/check/')
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    category = [key for key, _ in category.items()]

    def _update_backprop_weight(labels, fmask):
        '''
        誤差を伝播させる際の重みを指定．誤差を伝播させない部分は0．
        '''
        labels = labels.data.cpu().numpy()
        labels_y, labels_x = np.where(labels == 1)
        labels_y = np.append(labels_y, labels_y[-1] + 1)
        labels_x = np.append(labels_x, 0)

        weight = np.zeros((labels.shape[0] + 1, labels.shape[1]), int)
        row_p, columns_p = labels_y[0], [labels_x[0]]
        weight[row_p] = fmask[labels_x[0]]

        for row, column in zip(labels_y[1:], labels_x[1:]):
            if row == row_p:
                columns_p.append(column)
            else:
                if len(columns_p) > 1:
                    for y in columns_p:
                        weight[row_p][y] = 0

                row_p, columns_p = row, [column]

            weight[row] = weight[row] | fmask[column]

        weight = weight[:-1]
        weight = np.ones(labels.shape, int) - weight

        return weight

    # ---入力画像のタグから振り分け-----------------------------------------------
    outputs_path = path_top + 'outputs/check/confusion_matrix'

    counts = np.zeros((len(category), 2, 2))
    for image, label, _ in tqdm(imlist):
        fix_mask = _update_backprop_weight(label, mask)
        predicts = model.predict(image, labeling=True)
        for idx, flg in enumerate(fix_mask[0]):
            if flg == 0:
                continue

            if label[0][idx] == 0:
                # 正解が0のとき
                if predicts[0][idx] == 0:
                    # 予測が0であれば
                    counts[idx][0][0] += 1
                else:
                    # 予測が1であれば
                    counts[idx][1][0] += 1
            else:
                # 正解が1のとき
                if predicts[0][idx] == 0:
                    # 予測が0であれば
                    counts[idx][0][1] += 1
                else:
                    # 予測が1であれば
                    counts[idx][1][1] += 1

    DH.saveNpy(
        np.array(counts),
        '{0:0=2}_{1:0=2}'.format(int(threshold * 10), epoch),
        outputs_path
    )


def score_unknown(threshold=0.4, epoch=20):
    '''
    学習の際unknownとしている画像に対してどう予測しているかの確認
    '''
    # -------------------------------------------------------------------------
    # 準備
    import json
    import numpy as np
    import os
    import pickle
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import MultiLabelGCN
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_top = '../datas/gcn/'

    # maskの読み込み
    with open(path_top + 'inputs/comb_mask/{0:0=2}.pickle'.format(
              int(threshold * 10)), 'rb') as f:
        mask = pickle.load(f)

    # 誤差伝播の重みの読み込み
    with open(path_top + 'inputs/backprop_weight/{0:0=2}.npy'.format(
              int(threshold * 10)), 'rb') as f:
        bp_weight = np.load(f)

    # modelの設定
    gcn_settings = {
        'class_num': 3100,
        'filepaths': {
            'category': path_top + 'inputs/category.json',
            'upper_category': path_top + 'inputs/upper_category.json',
            'relationship': path_top + 'inputs/relationship.pickle',
            'learned_weight': path_top + 'inputs/learned/200cnn.pth'
        },
        'feature_dimension': 2048
    }
    model = MultiLabelGCN(
        class_num=3100,
        loss_function=MyLossFunction(),
        learningrate=0.1,
        weight_decay=1e-4,
        fix_mask=mask,
        network_setting=gcn_settings,
        multigpu=False,
        backprop_weight=bp_weight
    )
    if epoch > 0:
        model.loadmodel(
            '{0:0=3}weight'.format(epoch),
            path_top + 'outputs/learned/th{0:0=2}'.format(int(threshold * 10))
        )

    # データの読み込み
    imlist = DH.loadPickle('imlist', directory=path_top + 'outputs/check/')
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    category = [key for key, _ in category.items()]

    def _update_backprop_weight(labels, fmask):
        '''
        誤差を伝播させる際の重みを指定．誤差を伝播させない部分は1．
        '''
        labels = labels.data.cpu().numpy()
        labels_y, labels_x = np.where(labels == 1)
        labels_y = np.append(labels_y, labels_y[-1] + 1)
        labels_x = np.append(labels_x, 0)

        weight = np.zeros((labels.shape[0] + 1, labels.shape[1]), int)
        row_p, columns_p = labels_y[0], [labels_x[0]]
        weight[row_p] = fmask[labels_x[0]]

        for row, column in zip(labels_y[1:], labels_x[1:]):
            if row == row_p:
                columns_p.append(column)
            else:
                if len(columns_p) > 1:
                    for y in columns_p:
                        weight[row_p][y] = 0

                row_p, columns_p = row, [column]

            weight[row] = weight[row] | fmask[column]

        weight = weight[:-1]

        return weight

    # ---入力画像のタグから振り分け-----------------------------------------------
    outputs_path = path_top + 'outputs/check/unknown_predict'

    counts = np.zeros((len(category), 2))
    for image, label, _ in tqdm(imlist):
        fix_mask = _update_backprop_weight(label, mask)
        predicts = model.predict(image, labeling=True)
        for idx, flg in enumerate(fix_mask[0]):
            if flg == 0:
                continue

            if predicts[0][idx] == 0:
                # 予測が0であれば
                counts[idx][0] += 1
            else:
                # 予測が1であれば
                counts[idx][1] += 1

    DH.saveNpy(
        np.array(counts),
        '{0:0=2}_{1:0=2}'.format(int(threshold * 10), epoch),
        outputs_path
    )


def check_lowrecall(threshold=0.4):
    '''
    recallを元に画像や上位クラスとの関連を確認
    '''
    import json
    import numpy as np
    import torch
    from mmm import DataHandler as DH

    path_top = '../datas/gcn/'
    confusion_matrix = path_top + '_outputs/check/confusion_matrix/'
    confusion_matrix = DH.loadNpy(
        '{0:0=2}'.format(int(10 * threshold)), confusion_matrix
    )
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    local_df = DH.loadPickle(
        'local_df_area16_wocoth', '../datas/prepare/inputs'
    )
    w00 = torch.load(open(
        path_top + 'outputs3/learned/th{0:0=2}/000weight.pth'.format(
            int(10 * threshold)), 'rb')
    )
    w20 = torch.load(open(
        path_top + 'outputs3/learned/th{0:0=2}/020weight.pth'.format(
            int(10 * threshold)), 'rb')
    )

    results = []
    for cm, (cat, idx) in zip(confusion_matrix, category.items()):
        if cm[0][1] + cm[1][1] == 0:
            continue

        rc = cm[1][1] / (cm[0][1] + cm[1][1])
        if rc > 0.5:
            continue

        pr = 0 if sum(cm[1]) == 0 else cm[1][1] / sum(cm[1])
        results.append([
            cat,
            pr,
            rc,
            local_df.loc[cat]['representative'],
            # w00['layer1.W'][idx].cpu(),
            # w20['layer1.W'][idx].cpu()
        ])

    return np.array(results)


def confusion_all_matrix(threshold=0.4, epoch=20, saved=False):
    '''
    正例・unknown・負例についてconfusion_matrixを作成
    '''
    # -------------------------------------------------------------------------
    # 準備
    import json
    import numpy as np
    import os
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import MultiLabelGCN
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_top = '../datas/gcn/'

    # maskの読み込み
    mask = DH.loadPickle(
        '{0:0=2}'.format(int(threshold * 10)),
        path_top + 'inputs/comb_mask/'
    )

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy(
        '{0:0=2}'.format(int(threshold * 10)),
        path_top + 'inputs/backprop_weight/'
    )

    # modelの設定
    gcn_settings = {
        'num_class': 3100,
        'filepaths': {
            'category': path_top + 'inputs/category.json',
            'upper_category': path_top + 'inputs/upper_category.json',
            'relationship': path_top + 'inputs/relationship.pickle',
            'learned_weight': path_top + 'inputs/learned/200cnn.pth'
        },
        'feature_dimension': 2048
    }
    model = MultiLabelGCN(
        class_num=3100,
        loss_function=MyLossFunction(),
        learningrate=0.1,
        weight_decay=1e-4,
        fix_mask=mask,
        network_setting=gcn_settings,
        multigpu=False,
        backprop_weight=bp_weight
    )
    if epoch > 0:
        model.loadmodel(
            '{0:0=3}weight'.format(epoch),
            # path_top + 'outputs/learned/th{0:0=2}'.format(int(threshold * 10))
            path_top + 'outputs/learned_backup'
        )

    # データの読み込み
    imlist = DH.loadPickle('imlist_val', directory=path_top + 'outputs/check/')
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    category = [key for key, _ in category.items()]

    def _update_backprop_weight(labels, fmask):
        '''
        誤差を伝播させる際の重みを指定．誤差を伝播させない部分は0．
        '''
        labels = labels.data.cpu().numpy()
        labels_y, labels_x = np.where(labels == 1)
        labels_y = np.append(labels_y, labels_y[-1] + 1)
        labels_x = np.append(labels_x, 0)

        weight = np.zeros((labels.shape[0] + 1, labels.shape[1]), int)
        row_p, columns_p = labels_y[0], [labels_x[0]]
        weight[row_p] = fmask[labels_x[0]]

        for row, column in zip(labels_y[1:], labels_x[1:]):
            if row == row_p:
                columns_p.append(column)
            else:
                if len(columns_p) > 1:
                    for y in columns_p:
                        weight[row_p][y] = 0

                row_p, columns_p = row, [column]

            weight[row] = weight[row] | fmask[column]

        weight = weight[:-1]
        weight = np.ones(labels.shape, int) - weight

        return weight

    # ---入力画像のタグから振り分け-----------------------------------------------
    outputs_path = path_top + 'outputs/check/all_matrix'

    # 0: precision, 1: recall, 2: positive_1, 3: positive_all,
    # 4: unknown_1, 5: unknown_all, 6: negative_1, 7: negative_all
    counts = np.zeros((len(category), 8))

    for image, label, _ in tqdm(imlist):
        fix_mask = _update_backprop_weight(label, mask)
        predicts = model.predict(image, labeling=True)
        for idx, flg in enumerate(fix_mask[0]):
            # あるクラスcategory[idx]について
            if flg == 0:
                # 正解がunknownのとき
                if predicts[0][idx] == 1:
                    # 予測が1であれば
                    counts[idx][4] += 1

                continue

            if label[0][idx] == 0:
                # 正解が0のとき
                if predicts[0][idx] == 1:
                    # 予測が1であれば
                    counts[idx][6] += 1
            else:
                # 正解が1のとき
                if predicts[0][idx] == 1:
                    # 予測が1であれば
                    counts[idx][2] += 1

    allnum = len(imlist)
    for idx, (zero, one) in enumerate(bp_weight):
        counts[idx][3] = one
        counts[idx][5] = allnum - one - zero
        counts[idx][7] = zero

        if counts[idx][2] + counts[idx][6] != 0:
            counts[idx][0] = counts[idx][2] / (counts[idx][2] + counts[idx][6])
        if counts[idx][3] != 0:
            counts[idx][1] = counts[idx][2] / counts[idx][3]

    if saved:
        DH.saveNpy(
            np.array(counts),
            'all_matrix{0:0=2}_{1:0=2}'.format(int(threshold * 10), epoch),
            outputs_path
        )

    return np.array(counts)


def compare_pr(threshold=0.4, saved=True):
    '''
    トレーニング前後での精度の変化、各クラス、各クラスの上位クラス
    を一覧にしたリストの作成
    '''
    # -------------------------------------------------------------------------
    # 準備
    import json
    import numpy as np
    import os
    from mmm import DataHandler as DH
    from tqdm import tqdm

    path_top = '../datas/gcn/'

    outputs_path = path_top + 'outputs/check/all_matrix'
    epoch00 = 'all_matrix{0:0=2}_{1:0=2}.npy'.format(int(threshold * 10), 0)
    epoch20 = 'all_matrix{0:0=2}_{1:0=2}.npy'.format(int(threshold * 10), 20)
    if not os.path.isfile(outputs_path + '/' + epoch00):
        confusion_all_matrix(threshold, 0, True)
    if not os.path.isfile(outputs_path + '/' + epoch20):
        confusion_all_matrix(threshold, 20, True)

    epoch00 = DH.loadNpy(epoch00, outputs_path)
    epoch20 = DH.loadNpy(epoch20, outputs_path)
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    category = [key for key, _ in category.items()]
    upper_category = json.load(
        open(path_top + 'inputs/upper_category.json', 'r')
    )
    upper_category = [key for key, _ in upper_category.items()]
    local_df = DH.loadPickle(
        'local_df_area16_wocoth', '../datas/prepare/inputs'
    )
    # -------------------------------------------------------------------------
    # 0: label, 1: rep
    # 2 ~ 9: confusion_all_matrix of epoch 0
    # 10 ~ 17: confusion_all matrix of epoch 20

    compare_list = []
    for idx, cat in tqdm(enumerate(category)):
        if cat in upper_category:
            continue

        row = [cat, local_df.loc[cat]['representative']]
        row.extend(epoch00[idx])
        row.extend(epoch20[idx])

        compare_list.append(row)

    if saved:
        outputs_path = path_top + 'outputs/check/acc_change'
        DH.saveNpy(
            np.array(compare_list, dtype=object),
            'result_00to20_{0:0=2}'.format(int(threshold * 10)),
            outputs_path
        )

    return np.array(compare_list)


def check_locate():
    '''
    訓練画像の位置情報がデータセット内に正しく存在しているかどうかの確認
    '''
    # ---準備-------------------------------------------------------------------
    from mmm import DataHandler as DH
    from tqdm import tqdm

    path_top = '../datas/geo_rep/'
    photo_location = DH.loadPickle('photo_location_train', path_top + 'inputs')
    local_df = DH.loadPickle(
        '../datas/prepare/inputs/local_df_area16_wocoth2.pickle'
    )
    locate_list = []
    for geo in local_df.itertuples():
        locate_list.extend(list(geo.geo))

    # print(len(locate_list))
    # print(locate_list[0:5])
    imlist = DH.loadPickle('imlist', directory=path_top + 'inputs/')

    # ---チェック---------------------------------------------------------------
    cnt = 0
    for _, _, filename in tqdm(imlist):
        image_loc = photo_location[filename[0]]
        if image_loc not in locate_list:
            cnt += 1

    print(cnt, len(imlist), cnt / len(imlist))


def ite_hist_change(phase='train', width=0.16, saved=True):
    import matplotlib.pyplot as plt
    import numpy as np
    from mmm import DataHandler as DH

    plt.rcParams['font.family'] = 'IPAexGothic'
    # -------------------------------------------------------------------------
    dpath = '../datas/gcn/outputs/check/ite'
    data = DH.loadNpy('epoch00to20_thr04_{0}.npy'.format(phase), dpath)
    diff = [[] for _ in range(5)]

    for item in data:
        diff[len(item[1]) - 1].append(item[11] - item[3])

    bar_heights = np.zeros((5, 8))
    thresholds = np.arange(-1.0, 1.1, 0.25)

    for idx, item in enumerate(diff):
        bin_heights = np.histogram(item, bins=8, range=(-1.0, 1.0))[0]
        bar_heights[idx] = bin_heights / sum(bin_heights)

    # -------------------------------------------------------------------------
    # labels = ['{0:0.2f}'.format(item) for item in thresholds]
    x = np.arange(len(thresholds) - 1)
    width = width if 0 < width <= 0.2 else 0.16

    fig, ax = plt.subplots()

    ax.bar(x + 0.1 + 2.5 * width, np.ones(8), 5 * width, alpha=0.15)
    for idx, bh in enumerate(bar_heights):
        ax.bar(
            x + 0.5 + width * (idx - 2),
            bh, width,
            label='{0}個'.format(idx + 1)
        )

    x = np.arange(len(thresholds))
    labels = ['{0}'.format(int(item * 100)) for item in thresholds]

    fig_title = '学習用データセット' if phase == 'train' else '評価用データセット'
    ax.set_title(fig_title)
    ax.set_xlabel('学習前後での再現率の変化量(%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0 %', '25 %', '50 %', '75 %', '100 %'])
    ax.set_aspect(3)
    ax.legend(title='上位概念の数')

    plt.show()

    if saved:
        plt.savefig('../datas/gcn/outputs/check/ite/{0}.png'.format(phase))


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # r = get_predicts()
    # get_training_images(threshold=0.4)
    # class_precision(epoch=0)
    # score_unknown(epoch=0)
    # confusion_all_matrix(threshold=0.4, epoch=0, saved=True)
    confusion_all_matrix(threshold=0.4, epoch=20, saved=True)
    # r = check_lowrecall()
    # check_locate()
    # compare_pr(threshold=0.4, saved=True)
    # ite_hist_change('train')
    # ite_hist_change('validate')
    # -------------------------------------------------------------------------
    print('finish.')
