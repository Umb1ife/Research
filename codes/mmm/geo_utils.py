import numpy as np
from .datahandler import DataHandler as DH
from tqdm import tqdm


class GeoUtils:
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
    def rep_mask(category, sim_thr=5, reverse=True, saved=True,
                 save_path='../datas/geo_rep/inputs/',
                 base_path='../datas/bases/'):
        print('calculating mask ...')
        repsnum = len(category)
        _mask = np.zeros((repsnum, repsnum), int)
        sim_dict = DH.loadPickle('geo_rep_simdict', base_path)

        for tag1 in category:
            for tag2 in category:
                if tag1 == tag2:
                    continue

                sim = sim_dict[tag2][tag1] if reverse else sim_dict[tag1][tag2]
                if sim < sim_thr:
                    _mask[category[tag1]][category[tag2]] = 1

        if saved:
            DH.savePickle(_mask, 'mask_{0}'.format(sim_thr), save_path)

        return _mask

    @staticmethod
    def down_dataset(rep_category, local_category, phase='train',
                     base_path='../datas/bases/'):
        print('preparing dataset: {0} ...'.format(phase))
        lda = DH.loadPickle('local_df_area16_wocoth_new.pickle', base_path)
        down_category = sorted(list(set(local_category) - set(rep_category)))

        # datas = lda
        locates = 'geo' if phase == 'train' else 'geo_val'
        locates = list(lda[locates])

        locates = [item if len(item) >= 1 else [] for item in locates]
        tags = list(lda.index)
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
    def down_mask(rep_category, local_category, sim_thr=5, reverse=True,
                  saved=True, save_path='../datas/geo_down/inputs/',
                  base_path='../datas/bases/'):
        print('calculating mask ...')
        geo_category = {key: idx for idx, key in enumerate(local_category)}
        down_category = sorted(list(set(local_category) - set(rep_category)))
        num_classes = len(local_category)
        _mask = np.zeros((num_classes, num_classes), int)

        sim_dict = DH.loadPickle('geo_down_simdict', base_path)
        for tag1 in down_category:
            for tag2 in down_category:
                if tag1 == tag2:
                    continue

                sim = sim_dict[tag2][tag1] if reverse else sim_dict[tag1][tag2]
                if sim < sim_thr:
                    _mask[geo_category[tag1]][geo_category[tag2]] = 1

        if saved:
            DH.savePickle(_mask, 'mask_{0}'.format(sim_thr), save_path)

        return _mask
