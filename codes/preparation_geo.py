def make_photo_locate_dict(phase='train'):
    '''
    photo_loc_dict_201x.pickleから画像と位置情報をまとめたdictの作成
    '''
    from mmm import DataHandler as DH
    # -------------------------------------------------------------------------
    # パスの指定
    path_top = '../datas/prepare/'
    inputs_path = path_top + 'inputs'
    # outputs_path = path_top + 'outputs'
    outputs_path = '../datas/rr/inputs'

    # -------------------------------------------------------------------------
    filename = 'photo_loc_dict_2017.pickle' \
        if phase == 'train' else 'photo_loc_dict_2016.pickle'
    pldict = DH.loadPickle(filename, inputs_path)
    pldict = {
        '{0}.jpg'.format(photo): locate for photo, locate in pldict.items()
    }

    DH.savePickle(pldict, 'photo_location_' + phase, outputs_path)


def make_category(thr=100):
    import json
    from mmm import DataHandler as DH

    datas = '../datas/rr/inputs/local_df_area16_wocoth.pickle'
    datas = DH.loadPickle(datas)
    category = '../datas/gcn/inputs/category.json'
    category = json.load(open(category, 'rb'))
    category = [key for key, _ in category.items()]

    uppers = []
    visual_uppers = []
    for item in category:
        if item in datas.index:
            uppers.extend(list(datas.loc[item]['up']))
        else:
            visual_uppers.append(item)

    uppers = list(set(uppers))
    utags = [item for item in uppers if datas.loc[item]['max_freq'] >= thr]
    flags = utags.copy()
    top = ['arizona']
    # top = ['sanfrancisco']

    hierarchy = []
    while True:
        hierarchy.append(top)

        temp_top = []
        for tag in top:
            flags.remove(tag)
            temp_top.extend(datas.loc[tag]['down'])

        top = list(set(temp_top))
        top = [item for item in top if item in flags]

        if not top:
            break

    return hierarchy, flags


def remove_outlier(datas):
    from sklearn.ensemble import IsolationForest

    results = IsolationForest().fit_predict(datas)
    datas = [data for data, result in zip(datas, results) if result == 1]

    return datas


def make_imlist(phase='train', saved=True):
    import json
    import torch
    from mmm import DataHandler as DH
    from mmm import DatasetGeotag
    from tqdm import tqdm

    path_top = '../datas/rr/'
    category = json.load(open(path_top + 'inputs/category.json', 'rb'))
    num_class = len(category)
    kwargs_DF = {
        'train': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data_path': path_top
            + 'inputs/locate_dataset/finetune_train.pickle'
        },
        'validate': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data_path': path_top
            + 'inputs/locate_dataset/finetune_validate.pickle'
        },
    }

    dataset = DatasetGeotag(**kwargs_DF[phase])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    imlist = [(locate, label, 0) for locate, label, _ in tqdm(loader)]

    if saved:
        DH.savePickle(imlist, 'imlist', directory=path_top + 'inputs')

    return imlist


def make_backprop_ratio(sim_thr=0.4, saved=True):
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
    path_top = '../datas/rr/'

    # データの読み込み
    # imlist = DH.loadPickle(
    #     'imlist', directory=path_top + 'inputs/'
    # )
    imlist = make_imlist(phase='train', saved=False)
    category = json.load(open(path_top + 'inputs/category.json', 'r'))
    category = [key for key, _ in category.items()]
    class_num = len(category)

    # maskの読み込み
    mask = DH.loadPickle(
        # '{0:0=2}.pickle'.format(int(sim_thr * 10)),
        'thr_5.pickle',
        path_top + 'inputs/comb_mask_finetune'
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
    # for d, label, _ in tqdm(imlist):
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
        DH.saveNpy(np.array(counts), 'backprop_weight', path_top + 'inputs')

    return np.array(counts)


def make_geodataset(stage='finetune', phase='train', saved=True,
                    refined=False, thr=1):
    import numpy as np
    from mmm import DataHandler as DH
    from tqdm import tqdm

    datas = '../datas/rr/inputs/local_df_area16_wocoth.pickle'
    datas = DH.loadPickle(datas)
    mean, std = None, None
    locates = list(datas['geo'])
    temp = [item for gl in locates for item in gl]
    mean = np.mean(temp, axis=0)
    std = np.std(temp, axis=0)

    if refined:
        locates = [
            remove_outlier(item) if len(item) >= thr else []
            for item in tqdm(locates)
        ]

    locates = [item if len(item) >= thr else [] for item in locates]
    tags = list(datas.index)
    temp_dict = {key: [] for item in locates for key in item}
    for item, tag in zip(locates, tags):
        for locate in item:
            temp_dict[locate].append(tag)

    for key, val in temp_dict.items():
        temp_dict[key] = sorted(list(set(val)))

    hierarchy, flags = make_category()
    tags_dict = sorted(hierarchy[0] + hierarchy[1] + flags)
    tags_dict = DH.loadPickle('../datas/rr/inputs/geo_reps.pickle')
    # tags_dict = [
    #     'sanfrancisco',
    #     'oregon',
    #     'colorado',
    #     'utah',
    #     'newmexico',
    #     'michigan',
    #     'virginia',
    #     'massachusetts',
    #     'northcarolina',
    #     'georgia'
    # ]
    tags_dict.sort()
    tags_dict = {key: val for val, key in enumerate(tags_dict)}

    locate_tags_dictlist = []
    for key, val in temp_dict.items():
        temp = [tags_dict[label] for label in val if label in tags_dict]
        if temp:
            locate_tags_dictlist.append({
                'labels': temp,
                'locate': list(key)
            })

    if saved:
        output_path = '../datas/rr/inputs/'
        DH.savePickle(
            locate_tags_dictlist, '{0}_{1}.pickle'.format(stage, phase),
            output_path + 'locate_dataset/'
        )
        DH.saveJson(tags_dict, 'category.json', output_path)
        DH.saveNpy((mean, std), 'normalize_params', output_path)

    return {
        'locate_tags': locate_tags_dictlist,
        'tags': tags_dict,
        'norm': (mean, std)
    }


def _make_geodataset(stage='finetune', phase='train', saved=True,
                     normalized=True, refined=False, thr=100):
    import json
    import numpy as np
    from mmm import DataHandler as DH
    from tqdm import tqdm

    datas = '../datas/rr/inputs/local_df_area16_wocoth.pickle'
    datas = DH.loadPickle(datas)
    mean, std = None, None
    if normalized:
        temp = list(datas['geo'])
        temp = [item for gl in temp for item in gl]
        mean = np.mean(temp, axis=0)
        std = np.std(temp, axis=0)

        locates = []
        for locs in list(datas['geo']):
            temp = []
            for loc in locs:
                lon, lat = loc
                lon = (lon - mean[0]) / std[0]
                lat = (lat - mean[1]) / std[1]
                temp.append((lon, lat))
            locates.append(temp)
    else:
        locates = list(datas['geo'])

    if refined:
        locates = [
            remove_outlier(item) if len(item) >= thr else []
            for item in tqdm(locates)
        ]

    locates = [item if len(item) >= thr else [] for item in locates]
    tags = list(datas.index)
    temp_dict = {key: [] for item in locates for key in item}
    for item, tag in zip(locates, tags):
        for locate in item:
            temp_dict[locate].append(tag)

    for key, val in temp_dict.items():
        temp_dict[key] = sorted(list(set(val)))

    hierarchy, flags = make_category()
    tags_dict = sorted(hierarchy[0] + hierarchy[1] + flags)
    tags_dict = DH.loadPickle('../datas/rr/inputs/geo_reps.pickle')
    # tags_dict = [
    #     'sanfrancisco',
    #     'oregon',
    #     'colorado',
    #     'utah',
    #     'newmexico',
    #     'michigan',
    #     'virginia',
    #     'massachusetts',
    #     'northcarolina',
    #     'georgia'
    # ]
    tags_dict.sort()
    tags_dict = {key: val for val, key in enumerate(tags_dict)}

    locate_tags_dictlist = []
    for key, val in temp_dict.items():
        temp = [tags_dict[label] for label in val if label in tags_dict]
        if temp:
            locate_tags_dictlist.append({
                'labels': temp,
                'locate': list(key)
            })

    if saved:
        DH.savePickle(
            locate_tags_dictlist, '{0}_{1}.pickle'.format(stage, phase),
            '../datas/rr/inputs/locate_dataset/'
        )
        json.dump(tags_dict, open('../datas/rr/inputs/category.json', 'w'))

    return {
        'locate_tags': locate_tags_dictlist,
        'tags': tags_dict,
        'norm': (mean, std)
    }
    # return locate_tags_dictlist, tags_dict


def _make_mask(sim_thr=0.4, saved=True):
    import numpy as np
    import pandas as pd
    from mmm import DataHandler as DH

    dpath = '../datas/rr/inputs/'
    all_sim = DH.loadPickle('geo_rep_sim_03.pickle', dpath + 'geo/')
    all_sim = all_sim.values.tolist()
    category = DH.loadJson('category.json', dpath)

    left = [item[0][0] for item in all_sim]
    right = [item[0][1] for item in all_sim]
    sim = [item[1] for item in all_sim]

    comb_list = pd.DataFrame(
        zip(left, right, sim), columns=['left', 'right', 'sim']
    )
    comb_list = comb_list[
        (comb_list['left'].isin(category))
        & (comb_list['right'].isin(category))
    ]
    comb_list = comb_list[comb_list.sim >= sim_thr]
    comb_list = comb_list[['left', 'right']].values.tolist()

    comb_dict = list(set([item for tpl in comb_list for item in tpl]))
    comb_dict = {key: [] for key in comb_dict}
    for l, r in comb_list:
        comb_dict[l].append(r)
        comb_dict[r].append(l)

    # rep2index = {cat: idx for idx, cat in enumerate(category)}
    repsnum = len(category)
    ft_mask = np.zeros((repsnum, repsnum), int)

    for tag in category:
        if tag in comb_dict:
            for ctag in comb_dict[tag]:
                if ctag in category:
                    ft_mask[category[tag]][category[ctag]] = 1

    for i in range(repsnum):
        ft_mask[i, i] = 0

    if saved:
        DH.savePickle(
            ft_mask, '{0:0=2}.pickle'.format(int(sim_thr * 10)),
            dpath + 'comb_mask_finetune'
        )

    return ft_mask


def make_mask(sim_thr=5, saved=True, reverse=True):
    import numpy as np
    from mmm import DataHandler as DH

    dpath = '../datas/rr/inputs/'
    sim_dict = 'geo_rep_kl_03_dict_r' if reverse else 'geo_rep_kl_03_dict'
    sim_dict = DH.loadPickle(sim_dict, dpath + 'geo/')
    category = DH.loadJson('category.json', dpath)
    repsnum = len(category)
    mask = np.zeros((repsnum, repsnum), int)

    for tag1 in category:
        for tag2 in category:
            if tag1 == tag2:
                continue

            if sim_dict[tag1][tag2] < sim_thr:
                mask[category[tag1]][category[tag2]] = 1

    if saved:
        DH.savePickle(
            mask, 'thr_{0}.pickle'.format(sim_thr),
            dpath + 'comb_mask_finetune'
        )

    return mask


def get_geodatas(saved=True, savepath='../datas/rr/inputs/locate_dataset/',
                 stage='finetune', phase='train'):
    '''
    位置情報について上位クラスを取得
    '''
    import json
    import pickle
    from mmm import DataHandler

    filepath = '../datas/gcn/inputs/'
    annotation = json.load(
        open(filepath + '{0}_anno.json'.format(phase), 'rb')
    )
    photo_locates = pickle.load(open(
        '../datas/rr/inputs/photo_location_{0}.pickle'.format(phase), 'rb'
    ))

    geotag_dataset = []
    for dict_f in annotation:
        dict_f['locate'] = photo_locates[dict_f['file_name']]
        geotag_dataset.append(dict_f)

    filename = '{0}_{1}'.format(stage, phase)
    if saved:
        print('saved')
        DataHandler.savePickle(geotag_dataset, filename, savepath)

    return geotag_dataset


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # get_geodatas(savepath='../datas/rr/inputs/locate_dataset', phase='train')
    # make_geodataset(stage='finetune', phase='train', refined=False, thr=1)
    # make_mask(0.4)
    # make_imlist()
    make_backprop_ratio()
    # -------------------------------------------------------------------------
    print('finish.')
