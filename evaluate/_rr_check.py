def plot_map(stage='finetune', phase='train', refined=False,
             limited=None, sort_std=False):
    import colorsys
    import folium
    import numpy as np

    # -------------------------------------------------------------------------
    # from preparation_geo import make_geodataset
    # datas, category, (mean, std) = make_geodataset(
    #     stage=stage, phase=phase, saved=False, refined=refined, thr=1
    # ).values()

    from mmm import DataHandler as DH
    datas = DH.loadPickle('geo_down_train.pickle',
                          '../datas/small_graph/inputs/')
    category = DH.loadJson('geo_down_category.json', '../datas/small_graph/inputs/')
    mean, std = DH.loadNpy('normalize_params.npy', '../datas/small_graph/inputs')
    # -------------------------------------------------------------------------

    category = list(category.keys())
    class_num = len(category)

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

    HSV_tuples = [(x * 1.0 / class_num, 1.0, 1.0) for x in range(class_num)]
    RGB_tuples = [
        '#%02x%02x%02x' % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    ]

    for item in datas:
        labels, locate = item['labels'], item['locate']
        locate = [locate[1], locate[0]]
        radius = 150
        for lbl in labels:
            popup = category[lbl]
            if limited and popup not in limited:
                continue

            if sort_std:
                lbl = stds[lbl][1]

            folium.Circle(
                radius=radius,
                location=locate,
                popup=popup,
                color=RGB_tuples[lbl],
                fill=False,
            ).add_to(_map)
            radius *= 2

    return _map


def visualize_classmap(weight='../datas/geo_rep/outputs/learned/200weight.pth',
                       lat_range=(25, 50), lng_range=(-60, -125), unit=0.5,
                       limited=None):
    import colorsys
    import folium
    import numpy as np
    import torch
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import RepGeoClassifier

    # -------------------------------------------------------------------------
    # load classifier
    # from preparation_geo import make_geodataset
    # _, category, (mean, std) = make_geodataset(
    #     stage='finetune', phase='train', saved=False, refined=False, thr=1
    # ).values()

    from mmm import DataHandler as DH
    category = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    mean, std = DH.loadNpy('normalize_params.npy', '../datas/geo_rep/inputs')
    # -------------------------------------------------------------------------

    category = list(category.keys())
    num_class = len(category)
    model = RepGeoClassifier(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        # backprop_weight=bp_weight,
        network_setting={'class_num': num_class, 'mean': mean, 'std': std},
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
    HSV_tuples = [(x * 1.0 / num_class, 1.0, 1.0) for x in range(num_class)]
    RGB_tuples = [
        '#%02x%02x%02x' % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    ]

    # -------------------------------------------------------------------------
    # plot
    for lat in lats:
        # _lat = (lat - mean[1]) / std[1]
        for lng in lngs:
            # _lng = (lng - mean[0]) / std[0]
            # labels = model.predict(torch.Tensor([_lng, _lat]), labeling=True)
            labels = model.predict(torch.Tensor([lng, lat]), labeling=True)
            labels = np.where(labels > 0)[0]
            radius = 150
            for lbl in labels:
                popup = category[lbl]
                if limited and popup not in limited:
                    continue

                folium.Circle(
                    radius=radius,
                    location=[lat, lng],
                    popup=popup,
                    color=RGB_tuples[lbl],
                    fill=False
                ).add_to(_map)
                radius *= 2

    return _map


def visualize_pn(weight='../datas/geo_rep/outputs/learned/200cnn.pth',
                 limited=None, tag=''):
    import colorsys
    import folium
    import numpy as np
    import torch
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import RepGeoClassifier
    from preparation_geo import make_geodataset

    # -------------------------------------------------------------------------
    # load classifier
    datas, category, (mean, std) = make_geodataset(
        stage='finetune', phase='train', saved=False, refined=False, thr=1
    ).values()
    category = list(category.keys())

    num_class = len(category)
    model = RepGeoClassifier(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        network_setting={'class_num': num_class, 'mean': mean, 'std': std},
    )
    model.loadmodel(weight)

    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # make base map
    _map = folium.Map(
        location=[40.0, -100.0],
        zoom_start=4,
        tiles='Stamen Terrain'
    )

    # make colors list
    nc = 4
    HSV_tuples = [(x * 1.0 / nc, 1.0, 1.0) for x in range(nc)]
    RGB_tuples = [
        '#%02x%02x%02x' % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    ]
    RGB_tuples = ['red', 'blue', 'green', 'yellow']

    # -------------------------------------------------------------------------
    # plot
    for item in datas:
        labels, locate = item['labels'], item['locate']
        predicts = model.predict(torch.Tensor(locate), labeling=True)
        locate = [locate[1], locate[0]]
        radius = 150

        tags_t = [category[lbl] for lbl in labels]
        tags_p = [category[idx] for idx, flg in enumerate(predicts) if flg > 0]

        cid = 0 if tag in tags_p else 2
        cid = cid if tag in tags_t else cid + 1

        folium.Circle(
            radius=radius,
            location=locate,
            popup=str(cid),
            color=RGB_tuples[cid],
            fill=False
        ).add_to(_map)

    return _map


def confusion_all_matrix(
    epoch=200, saved=False,
    mask_path='../datas/geo_rep/inputs/comb_mask_finetune/thr_5.pickle',
    model_path='../datas/geo_rep/outputs/newnet_learned/',
    bpw_path='../datas/geo_rep/inputs/backprop_weight/thr5.npy',
    outputs_path='../datas/geo_rep/outputs/check/all_matrix/newnet.npy'
):
    '''
    正例・unknown・負例についてconfusion_matrixを作成
    '''
    # -------------------------------------------------------------------------
    # 準備
    import numpy as np
    import os
    from mmm import DataHandler as DH
    from mmm import RepGeoClassifier
    from preparation_geo import make_imlist
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_top = '../datas/geo_rep/'

    category = DH.loadJson('category.json', path_top + 'inputs/')
    num_class = len(category)

    # maskの読み込み
    mask = DH.loadPickle(mask_path)

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy(bpw_path)

    # modelの設定
    mean, std = DH.loadNpy('normalize_params', path_top + 'inputs')
    model = RepGeoClassifier(
        class_num=num_class,
        fix_mask=mask,
        backprop_weight=bp_weight,
        network_setting={'class_num': num_class, 'mean': mean, 'std': std},
    )
    if epoch > 0:
        model.loadmodel('{0}weight'.format(epoch), model_path)

    # データの読み込み
    # imlist = DH.loadPickle('imlist', directory=path_top + 'inputs')
    imlist = make_imlist(phase='train', saved=False)
    category = DH.loadJson('category.json', path_top + 'inputs')
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
        DH.saveNpy(np.array(counts), outputs_path)

    return np.array(counts)


def test(limited=None):
    # import numpy as np
    from mmm import MeanShiftDE
    from preparation_geo import make_geodataset
    from tqdm import tqdm

    datas, category, (mean, std) = make_geodataset(
        stage='finetune', phase='train', saved=False, normalized=True,
        refined=False, thr=100
    ).values()
    category = list(category.keys())
    # class_num = len(category)

    groups = {key: [] for key in category}
    for item in datas:
        for label in item['labels']:
            groups[category[label]].append(item['locate'])

    msde = MeanShiftDE()
    aaa = []
    for label, points in tqdm(groups.items()):
        tfs = [msde.is_positive(pnt, points, p=0.99) for pnt in points]

        aaa.append(sum(tfs) / len(tfs))
    # stds = [
    #     (idx, np.std(val)) for idx, (_, val) in enumerate(groups.items())
    # ]
    # stds.sort(key=lambda x: x[1])
    # stds = [(key, idx, val) for idx, (key, val) in enumerate(stds)]
    # stds.sort(key=lambda x: x[0])

    return aaa


def get_sample_image(tag, phase='train'):
    '''
    指定したタグの画像をfoliumで作成した地図上にプロット
    '''
    import base64
    import folium
    import glob
    import os
    from folium import IFrame
    from mmm import DataHandler as DH

    locate_dict = 'photo_location_train' if phase == 'train' \
        else 'photo_location_validate'
    locate_dict = DH.loadPickle(locate_dict, '../datas/geo_rep/inputs')
    phase = 'local_visual_2017' if phase == 'train' else 'local_visual_2016'
    image_path = '../datas/gcn/inputs/images/{0}/{1}/'.format(phase, tag)
    files = glob.glob(image_path + '*.jpg')

    _map = folium.Map(
        location=[40.0, -100.0],
        zoom_start=4,
        tiles='Stamen Terrain'
    )

    for img in files:
        lon, lat = locate_dict[os.path.basename(img)]
        encoded = base64.b64encode(open(img, 'rb').read())
        html = '<img src="data:image/jpg;base64,{}">'.format
        resolution, width, height = 10, 50, 25
        iframe = IFrame(
            html(encoded.decode('UTF-8')),
            width=(width * resolution) + 20,
            height=(height * resolution) + 20
        )
        popup = folium.Popup(iframe, max_width=1000)
        folium.Circle(
            radius=150,
            location=[lat, lon],
            popup=popup,
            color='red',
            fill=False,
        ).add_to(_map)

    return _map


def small_graph(tag, hops=2):
    from mmm import DataHandler as DH

    lda = '../datas/geo_rep/inputs/geo/local_df_area16_wocoth_kl5.pickle'
    lda = DH.loadPickle(lda)
    rda = '../datas/prepare/rep_df_area16_wocoth.pickle'
    rda = DH.loadPickle(rda)
    grd = '../datas/geo_rep/inputs/geo/geo_rep_df_area16_kl5.pickle'
    grd = DH.loadPickle(grd)
    grd = set(grd.index)

    # -------------------------------------------------------------------------
    # 指定したタグからn-hop以内のdownタグを取得
    layer = rda['down'][tag]
    down_tag, flgs = layer[:], []
    for _ in range(hops - 1):
        temp = []
        for item in layer:
            if item in flgs:
                continue

            flgs.append(item)
            temp.extend(lda['down'][item])

        temp = list(set(temp))
        down_tag.extend(temp)
        layer = temp[:]

    rda = set(rda.index)
    down_tag = sorted(list(set(down_tag) - rda - {tag}))
    down_tag = [
        item for item in down_tag if len(lda['geo_representative'][item]) > 0
    ]

    # -------------------------------------------------------------------------
    # 取得したdownタグからn-hop以内のvisual-repとなるタグを取得
    vis_rep = []
    layer = down_tag[:]
    flgs = []
    for _ in range(hops):
        temp = []
        for item in layer:
            if item in flgs or item in rda:
                continue

            flgs.append(item)
            temp.extend(lda['representative'][item])

        temp = list(set(temp))
        vis_rep.extend(temp)
        layer = temp[:]

    vis_rep = sorted(list((set(vis_rep) - set(down_tag)) & rda))

    # -------------------------------------------------------------------------
    # 取得したdownタグからn-hop以内のgeo-repとなるタグを取得
    geo_rep = []
    layer = down_tag[:]
    flgs = []
    for _ in range(hops):
        temp = []
        for item in layer:
            if item in flgs:
                continue

            flgs.append(item)
            temp.extend(lda['geo_representative'][item])

        temp = list(set(temp))
        geo_rep.extend(temp)
        layer = temp[:]

    geo_rep = sorted(list((set(geo_rep) - set(down_tag)) & grd))

    return vis_rep, geo_rep, down_tag


if __name__ == "__main__":
    # get_sample_image('1000islands')
    # # visualize_pn('../datas/geo_rep/outputs/newnet_learned/200weight.pth', tag='ca')
    # confusion_all_matrix(
    #     epoch=200, saved=True,
    #     mask_path='../datas/geo_rep/inputs/comb_mask_finetune/thr_6.pickle',
    #     model_path='../datas/geo_rep/outputs/newmask_6_learned/',
    #     bpw_path='../datas/geo_rep/inputs/backprop_weight/thr6.npy',
    #     outputs_path='../datas/geo_rep/outputs/check/all_matrix/newmask_6.npy'
    # )
    # plot_map()
    # test()
    # visualize_classmap()
    # plot_map()
    # small_graph('eagle')

    print('finish.')
