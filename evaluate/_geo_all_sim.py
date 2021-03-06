import os
import sys

sys.path.append(os.path.join('..', 'codes'))


import numpy as np
from mmm import DataHandler as DH
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


class GeoSim:
    def __init__(self, x_range=(-175, -64), y_range=(18, 71), fineness=(100, 50)):
        x_min, x_max = np.radians(sorted(x_range))
        y_min, y_max = np.radians(sorted(y_range))
        x = np.arange(x_min, x_max, (x_max - x_min) / fineness[0])
        y = np.arange(y_min, y_max, (y_max - y_min) / fineness[1])
        xx, yy = np.meshgrid(x, y)
        self._grid = np.c_[xx.ravel(), yy.ravel()]
        self._kde = KernelDensity(
            bandwidth=0.01,
            metric='euclidean',
            kernel='gaussian',
            algorithm='ball_tree'
        )

    def _cal_distribution(self, locate):
        self._kde.fit(np.radians(locate))
        distrib = np.exp(self._kde.score_samples(self._grid))
        # sum_dist = distrib.sum(keepdims=True)

        # return distrib / np.where(sum_dist == 0, 1, sum_dist)

        distrib += 1e-300
        distrib /= np.sum(distrib)

        return distrib

    def _data_shaping(self, lda):
        return [
            (idx, self._cal_distribution(data['geo']))
            for idx, data in tqdm(lda.iterrows(), total=lda.shape[0])
            if data['geo']
        ]

    def calculate_thr(self, lda, max_diff=10):
        lda = self._data_shaping(lda)
        all_sim_dict = {}
        for idx, (tag1, data1) in enumerate(tqdm(lda[:-1])):
            for tag2, data2 in lda[idx:]:
                d1 = entropy(data1, data2)
                d2 = entropy(data2, data1)

                if d1 > max_diff and d2 > max_diff:
                    continue

                if tag1 not in all_sim_dict:
                    all_sim_dict[tag1] = {}

                if tag2 not in all_sim_dict:
                    all_sim_dict[tag2] = {}

                all_sim_dict[tag1][tag2] = d2
                all_sim_dict[tag2][tag1] = d1

        return all_sim_dict

    def calculate_full(self, lda):
        all_sim_dict = {key: {} for key in lda.index}
        lda = self._data_shaping(lda)
        for idx, (tag1, data1) in enumerate(tqdm(lda[:-1])):
            for tag2, data2 in lda[idx:]:
                d1 = entropy(data1, data2)
                d2 = entropy(data2, data1)

                all_sim_dict[tag1][tag2] = d2
                all_sim_dict[tag2][tag1] = d1

        return all_sim_dict

    def _calculate(self, lda):
        all_sim_dict = {}
        for idx1, data1 in tqdm(lda.iterrows(), total=lda.shape[0]):
            if not data1['geo']:
                continue

            distrib1 = self._cal_distribution(data1['geo'])
            for idx2, data2 in lda.iterrows():
                if idx1 == idx2:
                    continue

                if not data2['geo']:
                    continue

                distrib2 = self._cal_distribution(data2['geo'])
                d1 = entropy(distrib1, distrib2)
                d2 = entropy(distrib2, distrib1)

                if d1 > 10 and d2 > 10:
                    continue

                if idx1 not in all_sim_dict:
                    all_sim_dict[idx1] = {}

                if idx2 not in all_sim_dict:
                    all_sim_dict[idx2] = {}

                all_sim_dict[idx1][idx2] = d1
                all_sim_dict[idx2][idx1] = d2

        return all_sim_dict


class GeoRepSim:
    def __init__(self, x_range=(-175, -64), y_range=(18, 71), fineness=(100, 50)):
        x_min, x_max = np.radians(sorted(x_range))
        y_min, y_max = np.radians(sorted(y_range))
        x = np.arange(x_min, x_max, (x_max - x_min) / fineness[0])
        y = np.arange(y_min, y_max, (y_max - y_min) / fineness[1])
        xx, yy = np.meshgrid(x, y)
        self._grid = np.c_[xx.ravel(), yy.ravel()]
        self._kde = KernelDensity(
            bandwidth=0.01,
            metric='euclidean',
            kernel='gaussian',
            algorithm='ball_tree'
        )

    def _cal_distribution(self, locate):
        self._kde.fit(np.radians(locate))
        distrib = np.exp(self._kde.score_samples(self._grid))
        # sum_dist = distrib.sum(keepdims=True)

        # return distrib / np.where(sum_dist == 0, 1, sum_dist)

        distrib += 1e-300
        distrib /= np.sum(distrib)

        return distrib

    def calculate(self, lda):
        grd = DH.loadPickle('../datas/bases/geo_rep_df_area16_kl5.pickle')
        sim_dict = {}
        for idx1, data1 in tqdm(grd.iterrows(), total=grd.shape[0]):
            distrib1 = self._cal_distribution(lda['geo'][idx1])
            for idx2, data2 in grd.iterrows():
                if idx1 == idx2:
                    continue

                distrib2 = self._cal_distribution(lda['geo'][idx2])
                d1 = entropy(distrib1, distrib2)
                d2 = entropy(distrib2, distrib1)

                if idx1 not in sim_dict:
                    sim_dict[idx1] = {}

                if idx2 not in sim_dict:
                    sim_dict[idx2] = {}

                sim_dict[idx1][idx2] = d2
                sim_dict[idx2][idx1] = d1

        return sim_dict


if __name__ == "__main__":
    lda_o = DH.loadPickle(
        '../datas/bases/local_df_area16_wocoth_new.pickle'
    )

    # asd = GeoSim().calculate_thr(lda_o)
    # asd = GeoSim().calculate_full(lda_o)
    # DH.saveJson(asd, 'geo_all_sim.json', '../datas/bases/')
    rsd = GeoRepSim().calculate(lda_o)
    DH.saveJson(rsd, 'geo_rep_simdict.json', '../datas/bases/')

    # grs = DH.loadJson('../datas/geo_rep/inputs/geo_rep_simdict.json')

    print('finish.')
