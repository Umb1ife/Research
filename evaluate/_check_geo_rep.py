import os
import sys

sys.path.append(os.path.join('..', 'codes'))


def plot_map(phase='train', refined=False, limited=None,
             saved=False, sort_std=False):
    import colorsys
    import folium
    import numpy as np
    from mmm import DataHandler as DH
    from mmm import GeoUtils as GU
    from tqdm import tqdm

    input_path = '../datas/geo_rep/inputs/'
    category = DH.loadJson('category.json', input_path)
    datas, (mean, std) = GU.rep_dataset(category, phase)
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
            radius += radius

    if saved:
        _map.save('../datas/geo_rep/outputs/check/georep_train.html')

    return _map


def classmap(weight='../datas/geo_rep/outputs/learned_232/200weight.pth',
             lat_range=(25, 50), lng_range=(-60, -125), unit=0.5,
             limited=None, saved=False, opacity=0.3, thr=0.5):
    import colorsys
    import folium
    import numpy as np
    import torch
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import RepGeoClassifier
    from tqdm import tqdm

    # -------------------------------------------------------------------------
    # load classifier
    category = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    category = list(category.keys())
    num_class = len(category)

    model = RepGeoClassifier(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        network_setting={
            'num_classes': num_class,
            'base_weight_path': '../datas/geo_base/outputs/learned_50x25/200weight.pth',
            'BR_settings': {'fineness': (50, 25)}
        }
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
    thr = min(max(0, thr), 0.999)
    for lat in tqdm(lats):
        for lng in lngs:
            labels = model.predict(torch.Tensor([lng, lat]))
            prd = model.predict(torch.Tensor([lng, lat]))
            labels = np.where(prd > thr)[0]
            prd = (prd - thr) / (1 - thr)
            radius = 30
            for lbl in labels:
                popup = category[lbl]
                if popup not in limited:
                    continue

                folium.Circle(
                    radius=radius,
                    location=[lat, lng],
                    popup=popup,
                    color=RGB_tuples[convert_idx[lbl]],
                    opacity=prd[lbl],
                    fill=False,
                ).add_to(_map)
                radius *= 2

    if saved:
        _map.save('../datas/geo_rep/outputs/check/georep_classmap.html')

    return _map


def visualize_classmap(weight='../datas/geo_rep/outputs/learned_232/200weight.pth',
                       lat_range=(25, 50), lng_range=(-60, -125), unit=0.5,
                       limited=None, saved=False, opacity=0.3):
    import colorsys
    import folium
    import numpy as np
    import torch
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import RepGeoClassifier
    from tqdm import tqdm

    # -------------------------------------------------------------------------
    # load classifier
    category = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    category = list(category.keys())
    num_class = len(category)

    model = RepGeoClassifier(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        network_setting={
            'num_classes': num_class,
            'base_weight_path': '../datas/geo_base/outputs/learned_50x25/200weight.pth',
            'BR_settings': {'fineness': (50, 25)}
        }
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
            labels = model.predict(torch.Tensor([lng, lat]), labeling=True)
            labels = np.where(labels > 0)[0]
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
                    opacity=opacity,
                    fill=True,
                    fill_opacity=opacity,
                ).add_to(_map)
                radius *= 2

    if saved:
        _map.save('../datas/geo_rep/outputs/check/georep_classmap.html')

    return _map


def confusion_all_matrix(epoch=200, saved=True,
                         outputs_path='../datas/geo_rep/outputs/check/'):
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
    from mmm import GeoUtils as GU
    from mmm import RepGeoClassifier
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # データの読み込み先
    input_path = '../datas/geo_rep/inputs/'

    category = DH.loadJson('category.json', input_path)
    num_class = len(category)

    geo_rep_train, (mean, std) = GU.rep_dataset(category, 'train')
    geo_rep_validate, _ = GU.rep_dataset(category, 'validate')

    kwargs_DF = {
        'train': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data': geo_rep_train
        },
        'validate': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data': geo_rep_validate
        },
    }

    train_dataset = DatasetGeotag(**kwargs_DF['train'])
    val_dataset = DatasetGeotag(**kwargs_DF['validate'])

    # maskの読み込み
    mask = GU.rep_mask(category, saved=False)

    # modelの設定
    model = RepGeoClassifier(
        class_num=num_class,
        momentum=0.9,
        fix_mask=mask,
        network_setting={
            'num_classes': num_class,
            'base_weight_path': '../datas/geo_base/outputs/learned/200weight.pth',
            'BR_settings': {'fineness': (20, 20)}
        }
    )
    model.loadmodel('../datas/geo_down/inputs/weights/nobp_za10.pth')

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


if __name__ == "__main__":
    # confusion_all_matrix(
    #     epoch=200,
    #     outputs_path='../datas/geo_rep/outputs/check/rep32_nobp/'
    # )
    # confusion_all_matrix(
    #     epoch=0,
    #     outputs_path='../datas/geo_rep/outputs/check/last/'
    # )
    # visualize_classmap(weight='../datas/geo_rep/outputs/learned_small/010weight.pth')
    # classmap(
    #     weight='../datas/geo_down/inputs/rep_weight.pth',
    #     unit=0.1, thr=0.7, saved=True
    # )
    # visualize_classmap(unit=0.5, saved=False)
    # plot_map(saved=True)

    print('finish.')
