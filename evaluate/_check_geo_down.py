import os
import sys

sys.path.append(os.path.join('..', 'codes'))


def plot_map(phase='train', limited=None, saved=False, sort_std=False):
    '''
    トレーニングデータのプロット
    '''
    import colorsys
    import folium
    import numpy as np
    from mmm import DataHandler as DH
    from mmm import GeoUtils as GU
    from tqdm import tqdm

    input_path = '../datas/geo_down/inputs/'
    rep_category = DH.loadJson('upper_category.json', input_path)
    category = DH.loadJson('category.json', input_path)
    datas = GU.down_dataset(rep_category, category, phase)
    category = list(category.keys())

    if sort_std:
        groups = {key: [] for key in category}
        for item in datas:
            for label in item['labels']:
                groups[category[label]].append(item['locate'])

        stds = [
            (idx, np.std(val)) for idx, (_, val) in enumerate(groups.items())
        ]
        stds.sort(key=lambda x: x[1])
        stds = [(key, idx, val) for idx, (key, val) in enumerate(stds)]
        stds.sort(key=lambda x: x[0])

    _map = folium.Map(
        location=[40.0, -100.0],
        zoom_start=4,
        tiles='Stamen Terrain'
    )

    limited = category[:] if limited is None else limited
    limited = set(limited) & set(category)
    convert_idx = {}
    cnt = 0
    for idx, cat in enumerate(category):
        if cat in limited:
            convert_idx[idx] = cnt
            cnt += 1

    color_num = len(convert_idx)
    HSV_tuples = [(x * 1.0 / color_num, 1.0, 1.0) for x in range(color_num)]
    RGB_tuples = [
        '#%02x%02x%02x' % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    ]

    for item in tqdm(datas):
        labels, locate = item['labels'], item['locate']
        locate = [locate[1], locate[0]]
        radius = 150
        for lbl in labels:
            popup = category[lbl]
            if popup not in limited:
                continue

            if sort_std:
                lbl = stds[lbl][1]

            folium.Circle(
                radius=radius,
                location=locate,
                popup=popup,
                color=RGB_tuples[convert_idx[lbl]],
                fill=True,
                fill_opacity=1
            ).add_to(_map)

    if saved:
        _map.save(
            '../datas/geo_down/outputs/check/geodown_{0}.html'.format(phase)
        )

    return _map


def visualize_classmap(weight='../datas/geo_down/outputs/learned_232/020weight.pth',
                       lat_range=(25, 50), lng_range=(-60, -125), unit=0.5,
                       limited=None, saved=True):
    '''
    指定した範囲内の点について予測クラスをプロット
    '''
    import colorsys
    import folium
    import numpy as np
    import torch
    from mmm import DataHandler as DH
    from mmm import GeotagGCN
    from tqdm import tqdm

    # -------------------------------------------------------------------------
    # load classifier
    rep_category = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    category = DH.loadJson('category.json', '../datas/geo_down/inputs')
    num_class = len(category)

    gcn_settings = {
        'category': category,
        'rep_category': rep_category,
        'relationship': DH.loadPickle('../datas/bases/geo_relationship.pickle'),
        'rep_weight': torch.load('../datas/geo_rep/outputs/learned_232/200weight.pth'),
        'base_weight': torch.load(
            '../datas/geo_base/outputs/learned_50x25/400weight.pth'
        ),
        'BR_settings': {'fineness': (50, 25)}
    }

    model = GeotagGCN(
        class_num=num_class,
        momentum=0.9,
        weight_decay=1e-4,
        network_setting=gcn_settings,
    )
    model.loadmodel(weight)

    # -------------------------------------------------------------------------
    # make points
    lat_range, lng_range = sorted(lat_range), sorted(lng_range)
    lats = np.arange(lat_range[0], lat_range[1], unit)
    lngs = np.arange(lng_range[0], lng_range[1], unit)

    # -------------------------------------------------------------------------
    # make base map
    _map = folium.Map(
        location=[40.0, -100.0],
        zoom_start=4,
        tiles='Stamen Terrain'
    )

    # make colors list
    category = list(category.keys())
    limited = category[:] if limited is None else limited
    limited = set(limited) & set(category)
    convert_idx = {}
    cnt = 0
    for idx, cat in enumerate(category):
        if cat in limited:
            convert_idx[idx] = cnt
            cnt += 1

    color_num = len(convert_idx)
    HSV_tuples = [(x * 1.0 / color_num, 1.0, 1.0) for x in range(color_num)]
    RGB_tuples = [
        '#%02x%02x%02x' % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    ]

    # -------------------------------------------------------------------------
    # plot
    for lat in tqdm(lats):
        for lng in lngs:
            # labels = model.predict(torch.Tensor([30, -80]), labeling=True)
            labels = model.predict(torch.Tensor([lng, lat]), labeling=True)
            labels = np.where(labels[0] > 0)[0]
            radius = 150
            for lbl in labels:
                popup = category[lbl]
                if popup not in limited:
                    continue

                folium.Circle(
                    radius=radius,
                    location=[lat, lng],
                    popup=popup,
                    color=RGB_tuples[convert_idx[lbl]],
                    opacity=0.5,
                    fill=False,
                ).add_to(_map)
                radius += radius

    if saved:
        _map.save('../datas/geo_down/outputs/check/geodown_predictmap.html')

    return _map


def visualize_training_data(tag, phase='train', limited=[-1, 0, 1]):
    '''
    あるクラスについて学習データを正例(1)・unknown(-1)・負例(0)に振り分けプロット
    '''
    import colorsys
    import folium
    import numpy as np
    import torch
    from mmm import DataHandler as DH
    from mmm import DatasetGeotag
    from mmm import GeoUtils as GU
    from tqdm import tqdm

    # -------------------------------------------------------------------------
    # データの読み込み
    input_path = '../datas/geo_down/inputs/'
    rep_category = DH.loadJson('upper_category.json', input_path)
    category = DH.loadJson('category.json', input_path)
    num_class = len(category)
    if tag not in category:
        raise Exception
    tag_idx = category[tag]

    geo_down_train = GU.down_dataset(rep_category, category, 'train')
    geo_down_validate = GU.down_dataset(rep_category, category, 'validate')

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

    dataset = DatasetGeotag(**kwargs_DF[phase])
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )

    mask = GU.down_mask(rep_category, category, sim_thr=5, saved=False)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # トレーニングデータを振り分け
    loc_ans_list = []
    for locate, label, _ in tqdm(loader):
        fix_mask = _fixed_mask(label, mask)
        locate = (float(locate[0][0]), float(locate[0][1]))
        flg = -1 if fix_mask[0][tag_idx] == 1 else 0 \
            if label[0][tag_idx] == 0 else 1

        loc_ans_list.append((locate, flg))

    # -------------------------------------------------------------------------
    # plot
    color_num = 3
    HSV_tuples = [(x * 1.0 / color_num, 1.0, 1.0) for x in range(color_num)]
    RGB_tuples = [
        '#%02x%02x%02x' % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    ]

    _map = folium.Map(
        location=[40.0, -100.0],
        zoom_start=4,
        tiles='Stamen Terrain'
    )

    print('plotting...')
    for locate, label in tqdm(loc_ans_list):
        if label not in limited:
            continue

        locate = [locate[1], locate[0]]
        folium.Circle(
            radius=150,
            location=locate,
            popup=label,
            color=RGB_tuples[label],
            fill=False,
        ).add_to(_map)

    return _map


def visualize_oneclass_predict(tag, phase='train', epoch=20,
                               weight_path='../datas/geo_down/outputs/learned_232',
                               limited=[0, 1], only_mistake=False):
    '''
    あるクラスについて学習データを正例(1)・unknown(-1)・負例(0)に振り分けプロット
    '''
    import folium
    import numpy as np
    import torch
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import DatasetGeotag
    from mmm import GeotagGCN
    from mmm import GeoUtils as GU
    from tqdm import tqdm

    # -------------------------------------------------------------------------
    # load classifier
    rep_category = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    category = DH.loadJson('category.json', '../datas/geo_down/inputs')
    num_class = len(category)

    gcn_settings = {
        'category': category,
        'rep_category': rep_category,
        'relationship': DH.loadPickle('../datas/bases/geo_relationship.pickle'),
        'rep_weight': torch.load('../datas/geo_rep/outputs/learned_232/200weight.pth'),
        'base_weight': torch.load(
            '../datas/geo_base/outputs/learned_50x25/400weight.pth'
        ),
        'BR_settings': {'fineness': (50, 25)}
    }

    model = GeotagGCN(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        weight_decay=1e-4,
        network_setting=gcn_settings,
    )
    model.loadmodel('{0:0=3}weight'.format(epoch), weight_path)

    # -------------------------------------------------------------------------
    # データの読み込み
    if tag not in set(category) - set(rep_category):
        raise Exception
    tag_idx = category[tag]

    geo_down_train = GU.down_dataset(rep_category, category, 'train')
    geo_down_validate = GU.down_dataset(rep_category, category, 'validate')

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

    dataset = DatasetGeotag(**kwargs_DF[phase])
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )

    mask = GU.down_mask(rep_category, category, sim_thr=5, saved=False)
    # mask = DH.loadPickle('../datas/geo_down/inputs/mask_5.pickle')

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # plot
    colors = [
        ('red', 'orange'),
        ('lightblue', 'blue'),
        ('lightgreen', 'darkgreen')
    ]

    _map = folium.Map(
        location=[40.0, -100.0],
        zoom_start=4,
        tiles='Stamen Terrain'
    )

    cnfmat = np.zeros(6)
    cmidx = {
        'label': {1: 1, -1: 3, 0: 5},
        'predict': {1: 0, -1: 2, 0: 4}
    }
    for locate, blabel, _ in tqdm(loader):
        fix_mask = _fixed_mask(blabel, mask)
        locate = (float(locate[0][0]), float(locate[0][1]))
        label = -1 if fix_mask[0][tag_idx] == 1 else 0 \
            if blabel[0][tag_idx] == 0 else 1

        predict = model.predict(torch.Tensor(locate), labeling=True)
        predict = predict[0][tag_idx]
        cnfmat[cmidx['label'][label]] += 1
        if predict == 1:
            cnfmat[cmidx['predict'][label]] += 1

        if label not in limited:
            continue

        locate = [locate[1], locate[0]]
        if label == predict or (label == -1 and predict == 1):
            if only_mistake:
                continue
        else:
            if only_mistake and label == -1:
                continue

        folium.Circle(
            radius=150,
            location=locate,
            popup=label,
            color=colors[label][int(predict)],
            fill=False,
        ).add_to(_map)

    print(cnfmat)
    print('recall: {0},  precision: {1}'.format(
        cnfmat[0] / cnfmat[1], cnfmat[0] / (cnfmat[0] + cnfmat[4])))
    return _map


def confusion_all_matrix(epoch=20, saved=True,
                         weight_path='../datas/geo_down/outputs/learned/',
                         outputs_path='../datas/geo_down/outputs/check/'):
    '''
    正例・unknown・負例についてconfusion_matrixを作成
    '''
    # -------------------------------------------------------------------------
    # 準備
    import numpy as np
    import os
    import torch
    from mmm import DataHandler as DH
    from mmm import DatasetGeotag
    from mmm import GeotagGCN
    from mmm import GeoUtils as GU
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # データの読み込み先
    base_path = '../datas/bases/'
    input_path = '../datas/geo_down/inputs/'

    rep_category = DH.loadJson('upper_category.json', input_path)
    category = DH.loadJson('category.json', input_path)
    num_class = len(category)

    geo_down_train = GU.down_dataset(
        rep_category, category, 'train',
        base_path=base_path
    )
    geo_down_validate = GU.down_dataset(
        rep_category, category, 'validate',
        base_path=base_path
    )

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

    # maskの読み込み
    mask = GU.down_mask(rep_category, category, reverse=False, saved=False)

    gcn_settings = {
        'category': category,
        'rep_category': rep_category,
        'relationship': DH.loadPickle('../datas/bases/geo_relationship.pickle'),
        'rep_weight': torch.load('../datas/geo_rep/outputs/learned_232/200weight.pth'),
        'base_weight': torch.load(
            '../datas/geo_base/outputs/learned_50x25/400weight.pth'
        ),
        'BR_settings': {'fineness': (50, 25)}
    }

    # modelの設定
    model = GeotagGCN(
        class_num=num_class,
        learningrate=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        fix_mask=mask,
        network_setting=gcn_settings,
    )
    model.loadmodel('{0:0=3}weight'.format(epoch), weight_path)

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
    # 0: precision, 1: recall, 2: positive_1, 3: positive_all,
    # 4: unknown_1, 5: unknown_all, 6: negative_1, 7: negative_all

    def count_result(dataset):
        from mmm import MakeBPWeight

        loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=1,
            num_workers=4
        )
        bp_weight = MakeBPWeight(dataset, len(category), mask)

        allnum = 0
        counts = np.zeros((len(category), 8))
        for locate, label, _ in tqdm(loader):
            allnum += 1
            fix_mask = _update_backprop_weight(label, mask)
            predicts = model.predict(locate, labeling=True)
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

        for idx, (zero, one) in enumerate(bp_weight):
            counts[idx][3] = one
            counts[idx][5] = allnum - one - zero
            counts[idx][7] = zero

            if counts[idx][2] + counts[idx][6] != 0:
                counts[idx][0] = counts[idx][2] / (counts[idx][2] + counts[idx][6])
            if counts[idx][3] != 0:
                counts[idx][1] = counts[idx][2] / counts[idx][3]

        return counts

    train_counts = count_result(train_dataset)
    validate_counts = count_result(val_dataset)

    if saved:
        DH.saveNpy(
            np.array(train_counts),
            'cm_train_{0:0=3}'.format(epoch),
            outputs_path
        )
        DH.saveNpy(
            np.array(validate_counts),
            'cm_validate_{0:0=3}'.format(epoch),
            outputs_path
        )

    return np.array(train_counts), np.array(validate_counts)


def check_change(filepath='../datas/geo_down/outputs/check/learned'):
    import numpy as np
    import os
    from mmm import DataHandler as DH

    ct000 = np.load(os.path.join(filepath, 'cm_train_000.npy'))
    ct020 = np.load(os.path.join(filepath, 'cm_train_020.npy'))
    # cv000 = np.load(ppp + '/cm_validate_000.npy')
    # cv020 = np.load(ppp + '/cm_validate_020.npy')
    category = DH.loadJson('../datas/geo_down/inputs/category.json')
    reps = DH.loadJson('../datas/geo_down/inputs/upper_category.json')
    gr = DH.loadPickle('../datas/bases/geo_relationship.pickle')

    downs = sorted(set(category) - set(reps))
    gr2 = {key: [] for idx, key in enumerate(downs)}
    for rep, down in gr.items():
        for item in down:
            if item not in downs:
                continue

            gr2[item].append(rep)

    concat = []
    for cat, bf, af in zip(category.keys(), ct000, ct020):
        if cat in reps:
            continue

        temp = [cat]
        temp.append(sorted(set(category) & set(gr2[cat])))
        temp.extend(bf)
        temp.extend(af)

        concat.append(temp)

    concat = np.array(concat)

    DH.saveNpy(concat, '../datas/geo_down/outputs/check/concat')

    return concat


if __name__ == "__main__":
    # check_change('../datas/geo_down/outputs/check/learned_232')
    # confusion_all_matrix(
    #     epoch=20,
    #     weight_path='../datas/geo_down/outputs/learned_232/',
    #     outputs_path='../datas/geo_down/outputs/check/learned_232/'
    # )
    # confusion_all_matrix(
    #     epoch=0,
    #     weight_path='../datas/geo_down/outputs/learned_232/',
    #     outputs_path='../datas/geo_down/outputs/check/learned_232/'
    # )
    # visualize_classmap()
    # visualize_classmap(
    #     weight='../datas/geo_down/outputs/learned_rep32_bp/020weight.pth',
    #     lat_range=(25, 50), lng_range=(-60, -125), unit=0.5, limited=['bellagio']
    # )
    # plot_map()
    # visualize_training_data('bellagio')
    # visualize_oneclass_predict('bellagio', only_mistake=True)

    print('finish.')
