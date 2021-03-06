import numpy as np
from .datahandler import DataHandler as DH
from tqdm import tqdm


class GeoUtils:
    @staticmethod
    def zerodata_augmentation(data, x_range=(-175, -64), y_range=(18, 71),
                              fineness=(20, 20), numdata_sqrt_oneclass=10):
        labels = set([i for i in range(fineness[0] * fineness[1])])

        # データのないブロックを取得
        x_min, x_max = sorted(x_range)
        y_min, y_max = sorted(y_range)
        xl, yl = (x_max - x_min) / fineness[0], (y_max - y_min) / fineness[1]
        for item in tqdm(data):
            x, y = item['locate']
            xlabel = (x - x_min) // xl
            ylabel = (y - y_min) // yl

            labels.discard(int(ylabel * fineness[0] + xlabel))

        # 全てのクラスに対し負例となるデータの追加
        for zerolabel in tqdm(labels):
            ylabel, xlabel = divmod(zerolabel, fineness[0])
            xgrid = np.linspace(
                x_min + xl * xlabel, x_min + xl * (xlabel + 1),
                numdata_sqrt_oneclass, endpoint=False
            )
            ygrid = np.linspace(
                y_min + yl * ylabel, y_min + yl * (ylabel + 1),
                numdata_sqrt_oneclass, endpoint=False
            )

            for x in xgrid:
                for y in ygrid:
                    data.append({'labels': [], 'locate': [x, y]})

        return data

    @staticmethod
    def base_dataset(x_range=(-175, -64), y_range=(18, 71), fineness=(20, 20),
                     numdata_sqrt_oneclass=32):
        print('preparing dataset for basenet ...')
        x_min, x_max = sorted(x_range)
        y_min, y_max = sorted(y_range)
        xgrid = np.arange(
            x_min, x_max,
            (x_max - x_min) / (fineness[0] * numdata_sqrt_oneclass)
        )
        ygrid = np.arange(
            y_min, y_max,
            (y_max - y_min) / (fineness[1] * numdata_sqrt_oneclass)
        )

        locate_tags_dictlist = []
        temp = []
        xl, yl = (x_max - x_min) / fineness[0], (y_max - y_min) / fineness[1]
        for x in tqdm(xgrid):
            xlabel = (x - x_min) // xl
            for y in ygrid:
                ylabel = (y - y_min) // yl

                locate_tags_dictlist.append({
                    'labels': [int(ylabel * fineness[0] + xlabel)],
                    'locate': [x, y]
                })
                temp.append([x, y])

        mean, std = np.mean(temp, axis=0), np.std(temp, axis=0)

        return locate_tags_dictlist, (mean, std)

    @staticmethod
    def rep_dataset(category, phase='train', base_path='../datas/bases/'):
        print('preparing dataset: {0} ...'.format(phase))
        lda = DH.loadPickle('local_df_area16_wocoth_new.pickle', base_path)
        locates = 'geo' if phase == 'train' else 'geo_val'
        locates = list(lda[locates])
        temp = [item for gl in locates for item in gl]
        mean = np.mean(temp, axis=0)
        std = np.std(temp, axis=0)

        locates = [item if len(item) >= 1 else [] for item in locates]
        tags = list(lda.index)
        temp_dict = {key: [] for item in locates for key in item}
        for item, tag in zip(locates, tags):
            for locate in item:
                temp_dict[locate].append(tag)

        for key, val in temp_dict.items():
            temp_dict[key] = sorted(list(set(val)))

        locate_tags_dictlist = []
        for key, val in tqdm(temp_dict.items()):
            temp = [category[label] for label in val if label in category]
            if temp:
                locate_tags_dictlist.append({
                    'labels': temp,
                    'locate': list(key)
                })

        return locate_tags_dictlist, (mean, std)

    @staticmethod
    def rep_mask(category, sim_thr=5, reverse=False, saved=True,
                 save_path='../datas/geo_rep/inputs/',
                 base_path='../datas/bases/'):
        print('calculating mask ...')
        repsnum = len(category)
        _mask = np.zeros((repsnum, repsnum), int)
        sim_dict = DH.loadJson('geo_rep_simdict', base_path)

        for tag1 in category:
            for tag2 in category:
                if tag1 == tag2:
                    continue

                sim = sim_dict[tag2][tag1] if reverse else sim_dict[tag1][tag2]
                if sim <= sim_thr:
                    _mask[category[tag1]][category[tag2]] = 1

        if saved:
            DH.savePickle(_mask, 'mask_{0}'.format(sim_thr), save_path)

        return _mask

    @staticmethod
    def down_dataset(rep_category, local_category, phase='train',
                     base_path='../datas/bases/'):
        print('preparing dataset: {0} ...'.format(phase))
        gda = DH.loadPickle('geospatial_df_area16_wocoth.pickle', base_path)
        down_category = sorted(list(set(local_category) - set(rep_category)))

        locates = 'geo' if phase == 'train' else 'geo_val'
        locates = list(gda[locates])

        locates = [item if len(item) >= 1 else [] for item in locates]
        tags = list(gda.index)
        temp_dict = {key: [] for item in locates for key in item}
        for item, tag in zip(locates, tags):
            for locate in item:
                temp_dict[locate].append(tag)

        for key, val in temp_dict.items():
            temp_dict[key] = sorted(list(set(val)))

        tags_dict = {key: val for val, key in enumerate(local_category)}

        locate_tags_dictlist = []
        for key, val in tqdm(temp_dict.items()):
            temp = [tags_dict[label] for label in val if label in down_category]
            if temp:
                locate_tags_dictlist.append({
                    'labels': temp,
                    'locate': list(key)
                })

        return locate_tags_dictlist

    @staticmethod
    def _down_mask(rep_category, local_category, sim_thr=5, reverse=False,
                   saved=True, save_path='../datas/geo_down/inputs/',
                   base_path='../datas/bases/'):
        print('calculating mask ...')
        geo_category = {key: idx for idx, key in enumerate(local_category)}
        down_category = sorted(list(set(local_category) - set(rep_category)))
        num_classes = len(local_category)
        _mask = np.zeros((num_classes, num_classes), int)

        sim_dict = DH.loadJson('geo_all_sim', base_path)
        for tag1 in tqdm(down_category):
            for tag2 in down_category:
                if tag1 == tag2:
                    continue

                sim = sim_dict[tag2][tag1] if reverse else sim_dict[tag1][tag2]
                if sim <= sim_thr:
                    _mask[geo_category[tag1]][geo_category[tag2]] = 1

        if saved:
            DH.savePickle(_mask, 'mask_{0}'.format(sim_thr), save_path)

        return _mask

    @staticmethod
    def down_mask(rep_category, local_category, sim_thr=5, reverse=False,
                  saved=True, save_path='../datas/geo_down/inputs/',
                  base_path='../datas/bases/'):
        print('calculating mask ...')
        all_sim = DH.loadJson('geo_all_sim.json', base_path)
        gsp_dict = DH.loadPickle('geospatial_df_area16_wocoth', base_path)
        gsp_dict = gsp_dict.to_dict('index')
        rep_dict = DH.loadPickle('geo_rep_df_area16_kl5', base_path)
        rep_dict = rep_dict.to_dict('index')

        comb_dict = {key: [] for key in local_category}
        for tag1 in local_category:
            for tag2 in local_category:
                if tag1 == tag2:
                    continue

                if all_sim[tag1][tag2] <= sim_thr:
                    comb_dict[tag1].append(tag2)

        # ---------------------------------------------------------------------
        lc = local_category
        down = sorted(list(set(lc) - set(rep_category)))
        tagsnum = len(lc)
        gspkeys = set(list(gsp_dict.keys()))
        lcset = set(lc)
        repset = set(rep_category)

        mask = np.zeros((tagsnum, tagsnum), int)
        for tag in tqdm(down):
            flgs = {tag}
            prev = set()
            checked = set()
            while flgs:
                for item in flgs:
                    checked.add(item)
                    prev = prev | set(gsp_dict[item]['geo_representative'])

                prev = prev & lcset
                flgs = (prev - checked) & gspkeys

            exprev = set()
            for ptag in prev:
                if ptag in comb_dict:
                    exprev = exprev | (set(comb_dict[ptag]) & repset)

            prev = prev | exprev

            for ptag in prev:
                mask[lc[tag]][lc[ptag]] = 1
                for ctag in comb_dict[ptag]:
                    mask[lc[tag]][lc[ctag]] = 1

                temp = set(rep_dict[ptag]['down']) & lcset
                for ttag in temp:
                    if ttag != tag:
                        mask[lc[tag]][lc[ttag]] = 1

                    if ttag in rep_dict:
                        ttagdown = set(rep_dict[ttag]['down']) & lcset
                        for tdtag in ttagdown:
                            if tdtag != tag:
                                mask[lc[tag]][lc[tdtag]] = 1

            if tag in rep_dict:
                for rtag in rep_dict[tag]['down']:
                    if rtag in lcset and rtag != tag:
                        mask[lc[tag]][lc[rtag]] = 1

            for ctag in comb_dict[tag]:
                mask[lc[tag]][lc[ctag]] = 1

        np.fill_diagonal(mask, 0)

        if saved:
            DH.savePickle(mask, 'mask_{0}'.format(sim_thr), save_path)

        return mask
