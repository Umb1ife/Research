def small_graph_tags(tag, lda, rda, grd_idx, hops=2):
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

    rda_idx = set(rda.index)
    down_tag = sorted(list(set(down_tag) - rda_idx - {tag}))
    down_tag = [
        item for item in down_tag if len(lda['geo_representative'][item]) > 0
    ]

    # -------------------------------------------------------------------------
    # 取得したdownタグからn-hop以内のvisual-repとなるタグを取得
    # vis_rep = []
    # layer = down_tag[:]
    # flgs = []
    # for _ in range(hops):
    #     temp = []
    #     for item in layer:
    #         if item in flgs or item in rda_idx:
    #             continue

    #         flgs.append(item)
    #         temp.extend(lda['representative'][item])

    #     temp = list(set(temp))
    #     vis_rep.extend(temp)
    #     layer = temp[:]

    # vis_rep = sorted(list((set(vis_rep) - set(down_tag)) & rda_idx))

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

    geo_rep = sorted(list((set(geo_rep) - set(down_tag)) & grd_idx))

    # return vis_rep, geo_rep, down_tag
    return tag, geo_rep, down_tag


def small_graph_pairs(sgt1, sgt2, lda, grd):
    tag1, tag2 = sgt1[0], sgt2[0]

    share_geo_rep = set(sgt1[1]) & set(sgt2[1])
    if not share_geo_rep:
        return [], []

    share_geo_down = []
    for item in share_geo_rep:
        share_geo_down.extend(grd['down'][item])
    share_geo_down = list(set(share_geo_down))

    tag1_down, tag2_down = [], []
    only_tag1_down, only_tag2_down = [], []
    for item in share_geo_down:
        if tag1 in lda['representative'][item]:
            tag1_down.append(item)
            if tag2 not in lda['representative'][item]:
                only_tag1_down.append(item)

        if tag2 in lda['representative'][item]:
            tag2_down.append(item)
            if tag1 not in lda['representative'][item]:
                only_tag2_down.append(item)

    return (tag1, tag2), tag1_down, tag2_down, only_tag1_down, \
        only_tag2_down, share_geo_rep, share_geo_down


def all_small_graph_pairs():
    from mmm import DataHandler as DH
    from tqdm import tqdm

    category = DH.loadJson('category.json', '../datas/fine_tuning/inputs')
    category = list(category.keys())
    lda = '../datas/rr/inputs/geo/local_df_area16_wocoth_kl5.pickle'
    lda = DH.loadPickle(lda)
    rda = '../datas/prepare/rep_df_area16_wocoth.pickle'
    rda = DH.loadPickle(rda)
    grd = '../datas/rr/inputs/geo/geo_rep_df_area16_kl5.pickle'
    grd = DH.loadPickle(grd)
    grd_idx = set(grd.index)

    # -------------------------------------------------------------------------
    pairs = []
    for cat1 in tqdm(category):
        temp = []
        for cat2 in category:
            if cat1 == cat2:
                continue

            sgt1 = small_graph_tags(cat1, lda, rda, grd_idx)
            sgt2 = small_graph_tags(cat2, lda, rda, grd_idx)
            sgp = small_graph_pairs(sgt1, sgt2, lda, grd)
            if sgp[1] and sgp[2]:
                temp.append((
                    (sgp[0][0], sgp[0][1]),
                    (
                        len(sgp[1]), len(sgp[2]), len(sgp[3]),
                        len(sgp[4]), len(sgp[5]), len(sgp[6])
                    ),
                    sgp[1], sgp[2], sgp[3], sgp[4], sgp[5], sgp[6]
                ))

        if temp:
            pairs.append(temp)

    return


def small_graph_pair(tag1, tag2, epochs=(200, 20, 200)):
    import glob
    import numpy as np
    import os
    import pandas as pd
    import shutil
    import torch
    import torch.backends.cudnn as cudnn
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import DatasetFlickr
    from mmm import DatasetGeotag
    from mmm import FinetuneModel
    from mmm import ImbalancedDataSampler as IDS
    from mmm import MultiLabelGCN
    from mmm import RepGeoClassifier
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import transforms
    from tqdm import tqdm

    category = DH.loadJson('category.json', '../datas/fine_tuning/inputs')
    category = list(category.keys())
    lda = '../datas/rr/inputs/geo/local_df_area16_wocoth_kl5.pickle'
    lda = DH.loadPickle(lda)
    rda = '../datas/prepare/rep_df_area16_wocoth.pickle'
    rda = DH.loadPickle(rda)
    grd = '../datas/rr/inputs/geo/geo_rep_df_area16_kl5.pickle'
    grd = DH.loadPickle(grd)
    grd_idx = set(grd.index)

    sgt1 = small_graph_tags(tag1, lda, rda, grd_idx)
    sgt2 = small_graph_tags(tag2, lda, rda, grd_idx)
    sgp = small_graph_pairs(sgt1, sgt2, lda, grd)

    data_path = '../datas/small_graph/'
    input_path = data_path + 'inputs/'
    vis_rep = sorted([tag1, tag2])
    geo_rep = sorted(sgp[5])
    down_category = sorted(sgp[3] + sgp[4])
    vis_local_tags = sorted(list(set(vis_rep) | set(down_category)))

    # -------------------------------------------------------------------------
    # 小規模グラフでのデータセット作成(visual)
    def vis_mask(sim_thr):
        all_sim = DH.loadPickle('all_sim.pickle', '../datas/prepare/')
        all_sim = all_sim.values.tolist()
        local_dict = lda.to_dict('index')
        represent_dict = rda.to_dict('index')

        left = [item[0][0] for item in all_sim]
        right = [item[0][1] for item in all_sim]
        sim = [item[1] for item in all_sim]
        representations = vis_rep
        down_tags = down_category
        local_tags = vis_local_tags

        comb_list = pd.DataFrame(
            zip(left, right, sim), columns=['left', 'right', 'sim']
        )
        comb_list = comb_list[
            (comb_list['left'].isin(local_tags))
            & (comb_list['right'].isin(local_tags))
        ]
        comb_list = comb_list[comb_list.sim >= sim_thr]
        comb_list = comb_list[['left', 'right']].values.tolist()

        comb_dict = list(set([item for tpl in comb_list for item in tpl]))
        comb_dict = {key: [] for key in comb_dict}
        for l, r in comb_list:
            comb_dict[l].append(r)
            comb_dict[r].append(l)
        # ---------------------------------------------------------------------
        # mask_の作成

        # for GCN
        tag2index = {cat: idx for idx, cat in enumerate(local_tags)}
        tagsnum = len(local_tags)
        gcn_mask = np.zeros((tagsnum, tagsnum), int)
        ldkeys = list(local_dict.keys())

        for tag in tqdm(down_tags):
            flgs = [tag]
            prev = []
            while flgs:
                temp = []
                for item in flgs:
                    temp.extend(list(
                        set(local_dict[item]['representative'])
                        & set(local_tags)
                    ))

                prev.extend(temp)
                flgs = list(set(temp) & set(ldkeys))

            addprev = []
            for ptag in prev:
                if ptag in comb_dict:
                    addprev.extend(list(
                        set(comb_dict[ptag]) & set(vis_rep)
                    ))

            prev = list(set(prev) | set(addprev))

            for ptag in prev:
                gcn_mask[tag2index[tag]][tag2index[ptag]] = 1
                for ctag in comb_dict[ptag]:
                    gcn_mask[tag2index[tag]][tag2index[ctag]] = 1

                temp = list(
                    set(represent_dict[ptag]['down']) & set(local_tags)
                )
                for ttag in temp:
                    if ttag != tag:
                        gcn_mask[tag2index[tag]][tag2index[ttag]] = 1

                    if ttag in represent_dict:
                        ttagdown = list(
                            set(represent_dict[ttag]['down'])
                            & set(local_tags)
                        )
                        for tdtag in ttagdown:
                            if tdtag != tag:
                                gcn_mask[tag2index[tag]][tag2index[tdtag]] = 1

            if tag in represent_dict:
                for rtag in represent_dict[tag]['down']:
                    if rtag in local_tags and rtag != tag:
                        gcn_mask[tag2index[tag]][tag2index[rtag]] = 1

            if tag in comb_dict:
                for ctag in comb_dict[tag]:
                    gcn_mask[tag2index[tag]][tag2index[ctag]] = 1

        for i in range(tagsnum):
            gcn_mask[i, i] = 0

        # for fine-tuning
        rep2index = {cat: idx for idx, cat in enumerate(representations)}
        repsnum = len(representations)
        ft_mask = np.zeros((repsnum, repsnum), int)

        for tag in representations:
            if tag in comb_dict:
                for ctag in comb_dict[tag]:
                    if ctag in representations:
                        ft_mask[rep2index[tag]][rep2index[ctag]] = 1

        for i in range(repsnum):
            ft_mask[i, i] = 0

        return ft_mask, gcn_mask

    def vis_annos(labels, stage, phase):
        photo_path = {
            'rep': {
                'train': '../datas/prepare/rep_train/',
                'validate': '../datas/prepare/rep_validate/'
            },
            'down': {
                'train': '../datas/gcn/inputs/images/local_visual_2017/',
                'validate': '../datas/gcn/inputs/images/local_visual_2016/'
            }
        }
        photo_path = photo_path[stage][phase]

        l2idx = vis_rep if stage == 'rep' else vis_local_tags
        l2idx = {key: idx for idx, key in enumerate(l2idx)}

        files = {}
        for lbl in labels:
            temp = glob.glob(photo_path + lbl + '/*.jpg')
            for photo in temp:
                photo = os.path.basename(photo)
                if photo in files:
                    files[photo]['labels'].append(l2idx[lbl])
                else:
                    files[photo] = {'file_name': photo, 'labels': [l2idx[lbl]]}

        return list(files.values())

    ft_mask, gcn_mask = vis_mask(0.4)
    if np.sum(ft_mask) != 0:
        raise Exception

    vat = vis_annos(vis_rep, 'rep', 'train')
    vav = vis_annos(vis_rep, 'rep', 'validate')
    vdat = vis_annos(down_category, 'down', 'train')
    vdav = vis_annos(down_category, 'down', 'validate')

    DH.savePickle(ft_mask, 'vis_ft_mask', input_path)
    DH.savePickle(gcn_mask, 'vis_gcn_mask', input_path)
    DH.saveJson(vat, 'vat.json', input_path)
    DH.saveJson(vav, 'vav.json', input_path)
    DH.saveJson(vdat, 'vdat.json', input_path)
    DH.saveJson(vdav, 'vdav.json', input_path)
    DH.saveJson(
        {key: idx for idx, key in enumerate(vis_rep)},
        'vis_rep_category.json', input_path
    )
    DH.saveJson(
        {key: idx for idx, key in enumerate(vis_local_tags)},
        'vis_down_category.json', input_path
    )
    # -------------------------------------------------------------------------
    # 初期設定
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    batchsize = 16
    numwork = 4

    # -------------------------------------------------------------------------
    # vis_fine_tuning
    image_path = '../datas/fine_tuning/inputs/images/'
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    kwargs_DF = {
        'train': {
            'filenames': {
                'Annotation': input_path + 'vat.json',
                'Category_to_Index': input_path + 'vis_rep_category.json'
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
            'image_path': image_path + 'train/'
        },
        'validate': {
            'filenames': {
                'Annotation': input_path + 'vav.json',
                'Category_to_Index': input_path + 'vis_rep_category.json'
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
            'image_path': image_path + 'validate/'
        }
    }

    train_dataset = DatasetFlickr(**kwargs_DF['train'])
    val_dataset = DatasetFlickr(**kwargs_DF['validate'])
    num_class = train_dataset.num_category()

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

    # modelの設定
    model = FinetuneModel(
        class_num=num_class,
        learningrate=0.01,
        momentum=0.9,
    )

    log_dir = data_path + 'log/fine_tuning'
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    print('log -> {0}'.format(log_dir))
    writer = SummaryWriter(log_dir=log_dir)

    model.savemodel('000cnn.pth', data_path + 'outputs')
    for epoch in range(epochs[0]):
        train_loss, train_recall, train_precision, _, _, _ \
            = model.train(train_loader)
        val_loss, val_recall, val_precision, _, _, _ \
            = model.validate(val_loader)

        print(
            'epoch %d, loss: %.4f val_loss: %.4f train_recall: %.4f \
                val_recall: %.4f train_precision: %.4f val_precision: %.4f'
            % (
                epoch, train_loss, val_loss, train_recall,
                val_recall, train_precision, val_precision
            )
        )

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

    model.savemodel('200cnn.pth', data_path + 'outputs')
    writer.close()
    # -------------------------------------------------------------------------
    # vis_gcn
    image_path = '../datas/gcn/inputs/images/'
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    kwargs_DF = {
        'train': {
            'filenames': {
                'Annotation': input_path + 'vdat.json',
                'Category_to_Index': input_path + 'vis_down_category.json'
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
            'image_path': image_path + 'train/'
        },
        'validate': {
            'filenames': {
                'Annotation': input_path + 'vdav.json',
                'Category_to_Index': input_path + 'vis_down_category.json'
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
            'image_path': image_path + 'validate/'
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

    # 誤差伝播の重みの設定
    def make_backprop_ratio(category_length, savename, _mask):
        '''
        GCNのトレーニングの際，正例が負例に対し極端に少なくなることに対し，
        誤差伝播の重みを変えることで対応するための割合の取得．
        '''

        # ---------------------------------------------------------------------
        # データの読み込み
        loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=1,
            num_workers=numwork
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
        counts = [[0, 0] for _ in range(category_length)]
        for _, label, _ in tqdm(loader):
            fix_mask = _fixed_mask(label, _mask)
            for idx, flg in enumerate(fix_mask[0]):
                if flg == 1:
                    continue

                if label[0][idx] == 0:
                    counts[idx][0] += 1
                else:
                    counts[idx][1] += 1

        DH.saveNpy(np.array(counts), savename, input_path)
        return np.array(counts)

    bp_weight = make_backprop_ratio(
        len(vis_local_tags), 'vis_down_bp_weight', gcn_mask
    )

    # 学習で用いるデータの設定や読み込み先
    gcn_settings = {
        'num_class': num_class,
        'filepaths': {
            'category': input_path + 'vis_down_category.json',
            'upper_category': input_path + 'vis_rep_category.json',
            'relationship': input_path + 'vis_relationship.pickle',
            'learned_weight': data_path + 'outputs/200cnn.pth'
        },
        'feature_dimension': 2048
    }

    # modelの設定
    model = MultiLabelGCN(
        class_num=num_class,
        learningrate=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        fix_mask=gcn_mask,
        network_setting=gcn_settings,
        backprop_weight=bp_weight
    )

    log_dir = data_path + 'log/gcn'
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    print('log -> {0}'.format(log_dir))
    writer = SummaryWriter(log_dir=log_dir)

    model.savemodel('00weight', data_path + 'outputs')
    for epoch in range(epochs[1]):
        train_loss, train_recall, train_precision, _, _, _ \
            = model.train(train_loader)
        val_loss, val_recall, val_precision, _, _, _ \
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

    model.savemodel('20weight', data_path + 'outputs')
    writer.close()

    # -------------------------------------------------------------------------
    # 小規模グラフでのデータセット作成(geo)
    def geo_dataset(phase='train'):
        datas = '../datas/rr/inputs/local_df_area16_wocoth.pickle'
        datas = DH.loadPickle(datas)
        locates = list(datas['geo'])
        temp = [item for gl in locates for item in gl]
        mean = np.mean(temp, axis=0)
        std = np.std(temp, axis=0)

        locates = [item if len(item) >= 1 else [] for item in locates]
        tags = list(datas.index)
        temp_dict = {key: [] for item in locates for key in item}
        for item, tag in zip(locates, tags):
            for locate in item:
                temp_dict[locate].append(tag)

        for key, val in temp_dict.items():
            temp_dict[key] = sorted(list(set(val)))

        tags_dict = {key: val for val, key in enumerate(geo_rep)}

        locate_tags_dictlist = []
        for key, val in temp_dict.items():
            temp = [tags_dict[label] for label in val if label in tags_dict]
            if temp:
                locate_tags_dictlist.append({
                    'labels': temp,
                    'locate': list(key)
                })

        return locate_tags_dictlist, tags_dict, (mean, std)

    def geo_mask(sim_thr=5, reverse=True):
        sim_dict = '../datas/rr/inputs/geo/geo_rep_kl5_kl_dict'
        sim_dict = sim_dict + '_r' if reverse else sim_dict
        sim_dict = DH.loadPickle(sim_dict)
        geo_rep_category = {key: idx for idx, key in enumerate(geo_rep)}
        repsnum = len(geo_rep_category)
        _mask = np.zeros((repsnum, repsnum), int)

        for tag1 in geo_rep_category:
            for tag2 in geo_rep_category:
                if tag1 == tag2:
                    continue

                if sim_dict[tag1][tag2] < sim_thr:
                    _mask[geo_rep_category[tag1]][geo_rep_category[tag2]] = 1

        return _mask

    geo_rep_train, geo_rep_category, (mean, std) = geo_dataset('train')
    geo_rep_validate, _, _ = geo_dataset('validate')
    geo_rep_mask = geo_mask(5, True)

    DH.savePickle(geo_rep_train, 'geo_rep_train', input_path)
    DH.savePickle(geo_rep_validate, 'geo_rep_validate', input_path)
    DH.saveJson(geo_rep_category, 'geo_rep_category', input_path)
    DH.saveNpy((mean, std), 'rep_normalize_params', input_path)
    DH.savePickle(geo_rep_mask, 'geo_rep_mask', input_path)

    # -------------------------------------------------------------------------
    # geo_rep
    num_class = len(geo_rep)
    kwargs_DF = {
        'train': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data_path': input_path + 'geo_rep_train.pickle'
        },
        'validate': {
            'class_num': num_class,
            'transform': torch.tensor,
            'data_path': input_path + 'geo_rep_validate.pickle'
        },
    }

    train_dataset = DatasetGeotag(**kwargs_DF['train'])
    val_dataset = DatasetGeotag(**kwargs_DF['validate'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=numwork
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=numwork
    )

    if torch.cuda.is_available():
        train_loader.pin_memory = True
        val_loader.pin_memory = True
        cudnn.benchmark = True

    bp_weight = make_backprop_ratio(
        num_class, 'geo_rep_bp_weight', geo_rep_mask
    )

    model = RepGeoClassifier(
        class_num=num_class,
        loss_function=MyLossFunction(reduction='none'),
        learningrate=0.1,
        momentum=0.9,
        fix_mask=geo_rep_mask,
        multigpu=False,
        backprop_weight=bp_weight,
        network_setting={'class_num': num_class, 'mean': mean, 'std': std},
    )

    log_dir = data_path + 'log/geo_rep'
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    print('log -> {0}'.format(log_dir))
    writer = SummaryWriter(log_dir=log_dir)

    model.savemodel('rep_000cnn.pth', data_path + 'outputs')
    for epoch in range(epochs[2]):
        train_loss, train_recall, train_precision, _, _, _ \
            = model.train(train_loader)
        val_loss, val_recall, val_precision, _, _, _ \
            = model.validate(val_loader)

        print(
            'epoch %d, loss: %.4f val_loss: %.4f train_recall: %.4f \
                val_recall: %.4f train_precision: %.4f val_precision: %.4f'
            % (
                epoch, train_loss, val_loss, train_recall,
                val_recall, train_precision, val_precision
            )
        )

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

    model.savemodel('rep_200cnn.pth', data_path + 'outputs')
    writer.close()


if __name__ == "__main__":
    small_graph_pair('airplane', 'castle', (20, 20, 200))

    print('finish.')
