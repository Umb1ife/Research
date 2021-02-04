import numpy as np
import pandas as pd
from mmm import DataHandler as DH
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


kde = KernelDensity(
    bandwidth=0.01,
    metric='euclidean',
    kernel='gaussian',
    algorithm='ball_tree'
)
x_min = -175 * np.pi / 180.
x_max = -64 * np.pi / 180.
y_min = 18 * np.pi / 180.
y_max = 71 * np.pi / 180.
x = np.arange(x_min, x_max, (x_max - x_min) / 100)
y = np.arange(y_min, y_max, (y_max - y_min) / 50)
xx, yy = np.meshgrid(x, y)
grid = np.c_[xx.ravel(), yy.ravel()]


def density_kl_divergence2(X1, X2, epsilon=1e-300):
    X1_rad = np.array([[x[0], x[1]] for x in X1]) * np.pi / 180.
    X2_rad = np.array([[x[0], x[1]] for x in X2]) * np.pi / 180.
    kde.fit(X1_rad)
    Z1 = np.exp(kde.score_samples(grid))
    kde.fit(X2_rad)
    Z2 = np.exp(kde.score_samples(grid))
    Z1 += epsilon
    Z2 += epsilon
    Z1 /= np.sum(Z1)
    Z2 /= np.sum(Z2)

    return entropy(Z2, Z1), entropy(Z1, Z2)


def density_kl_divergence(X1, X2):
    X1_rad = np.array([[x[0], x[1]] for x in X1]) * np.pi / 180.
    X2_rad = np.array([[x[0], x[1]] for x in X2]) * np.pi / 180.

    kde.fit(X1_rad)
    Z1 = np.exp(kde.score_samples(grid))
    kde.fit(X2_rad)
    Z2 = np.exp(kde.score_samples(grid))

    sumZ1, sumZ2 = Z1.sum(keepdims=True), Z2.sum(keepdims=True)
    Z1 /= np.where(sumZ1 == 0, 1, sumZ1)
    Z2 /= np.where(sumZ2 == 0, 1, sumZ2)

    # return entropy(Z2, Z1), entropy(Z1, Z2)
    return entropy(Z1, Z2), entropy(Z2, Z1)


# original
# local_df = pd.read_pickle('2017_usa/local_df_area16_wocoth.pickle')
# local_df = DH.loadPickle('../datas/bases/local_df_area16_wocoth_new.pickle')
# refined
local_df = DH.loadPickle('../datas/geo_down/inputs/local_df_area16_wocoth_new.pickle')
local_dict = local_df.to_dict('index')

geo_sim_list = []
for down in tqdm(local_dict):
    X1 = local_dict[down]['geo']
    if not X1:
        continue

    candid = local_dict[down]['up']
    checked = [down]
    while (candid):
        candid2 = []
        for up in candid:
            checked.append(up)
            X2 = local_dict[up]['geo']
            if not X2:
                continue

            sim1, sim2 = density_kl_divergence(X1, X2)
            if (sim1 <= 10 or sim2 <= 10):
                geo_sim_list.append((
                    down,
                    up,
                    sim1,
                    # sim2,
                    local_dict[down]['total_freq'],
                    local_dict[up]['total_freq']
                ))
                candid2.extend(local_dict[up]['up'])

        candid = list(set(candid2) - set(checked))

geo_sim_df = pd.DataFrame(geo_sim_list, columns=['down', 'up', 'sim', 'down_freq', 'up_freq'])
co_tag_list = [(x[0], x[1]) for x in geo_sim_list]
geo_sim_df['co_tag'] = co_tag_list
geo_sim_dict = geo_sim_df.set_index('co_tag').to_dict('index')
for tag in local_dict:
    local_dict[tag]['geo_representative'] = []

# get representative tag
# visual_df = pd.read_pickle('2017_usa/visual_df_area16_wocoth_new.pickle')
# visual_df = DH.loadPickle('../datas/prepare/inputs/visual_df_area16_wocoth.pickle')
visual_df = DH.loadPickle('../datas/geo_down/inputs/visual_df_area16_wocoth.pickle')
visual_dict = visual_df.to_dict('index')
base_visual_list = list(set(visual_dict.keys()) & set(list(local_dict.keys())))
for tag in tqdm(base_visual_list):
    checked = [tag]
    candid = local_dict[tag]['up']
    while (candid):
        candid2 = []
        for co_tag in candid:
            checked.append(co_tag)
            if (tag, co_tag) not in geo_sim_dict:
                continue

            if geo_sim_dict[(tag, co_tag)]['sim'] > 5.0:
                continue

            flag = 0
            for co_tag2 in local_dict[co_tag]['up']:
                if co_tag2 == tag:
                    continue

                if (
                    (tag, co_tag2) in geo_sim_dict
                    and geo_sim_dict[(tag, co_tag2)]['sim'] <= 5.0
                    and (co_tag, co_tag2) in geo_sim_dict
                    and geo_sim_dict[(co_tag, co_tag2)]['sim'] <= 5.0
                    and co_tag not in local_dict[co_tag2]['up']
                ):
                    flag = 1
                    if co_tag2 not in checked and co_tag2 not in candid:
                        candid2.append(co_tag2)

            if flag == 0:
                local_dict[tag]['geo_representative'].append(co_tag)

        candid = list(set(candid2))


DH.savePickle(local_dict, 'local_df_area16_wocoth_refined_r.pickle')
print('finish.')
