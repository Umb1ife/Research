import pandas as pd
from mmm import DataHandler as DH
from mmm import VisUtils as VU
from tqdm import tqdm


def geodata_format():
    local_dict = DH.loadPickle(
        '../datas/backup/def_georep/local_df_area16_wocoth_refined_r.pickle'
    )
    lda_new = pd.DataFrame(local_dict).T
    lda_data = DH.loadPickle(
        '../datas/geo_down/inputs/local_df_area16_wocoth_new.pickle'
    )

    for idx, data in tqdm(lda_data.iterrows(), total=lda_data.shape[0]):
        lda_new.at[idx, 'geo'] = data['geo']
        lda_new.at[idx, 'geo_val'] = data['geo_val']

    DH.savePickle(
        lda_new,
        'local_df_area16_wocoth_refined.pickle',
        '../datas/geo_down/inputs/'
    )


def vis_tag2file(stage='rep', saved=True):
    import os

    train = '../datas/bases/{0}_train/'.format(stage)
    files = os.listdir(train)
    train_tag = [f for f in files if os.path.isdir(os.path.join(train, f))]
    validate = '../datas/bases/{0}_validate/'.format(stage)
    files = os.listdir(validate)
    validate_tag = [f for f in files if os.path.isdir(os.path.join(validate, f))]

    tags = set(train_tag) & set(validate_tag)
    tag2file = {'train': {}, 'validate': {}}
    for tag in tags:
        tag2file['train'][tag] = {
            'filename': os.listdir(train + tag + '/'),
            'num': len(os.listdir(train + tag + '/'))
        }
        tag2file['validate'][tag] = {
            'filename': os.listdir(validate + tag + '/'),
            'num': len(os.listdir(validate + tag + '/'))
        }

    if saved:
        DH.saveJson(tag2file, '../datas/bases/{0}_tag2file.json'.format(stage))

    return tag2file


def vis_sim_dict(saved=True):
    simdf = DH.loadPickle('all_sim.pickle', '../datas/backup/prepare/')
    sim_dict = {}
    for _, ((l, r), s) in tqdm(simdf.iterrows(), total=simdf.shape[0]):
        if l not in sim_dict:
            sim_dict[l] = {}

        if r not in sim_dict:
            sim_dict[r] = {}

        sim_dict[l][r] = s
        sim_dict[r][l] = s

    if saved:
        DH.saveJson(sim_dict, 'vis_sim_dict', '../datas/bases/')

    return sim_dict


def samedown():
    from geodown_training import limited_category

    t2f = DH.loadJson('../datas/bases/down_tag2file.json')
    t2f = set(t2f['train'].keys())
    visrep = DH.loadJson('../datas/vis_rep/inputs/upper_category.json')
    # visdown_old = DH.loadJson('../datas/gcn/inputs/category.json')
    georep = DH.loadJson('../datas/geo_rep/inputs/category.json')
    geodown = limited_category(georep)

    down = set(geodown) - set(georep)
    down = down & t2f
    visdown = down | set(visrep)
    visdown = sorted(list(visdown))
    visdown = {cat: idx for idx, cat in enumerate(visdown)}
    geodown = down | set(georep)
    geodown = sorted(list(geodown))
    geodown = {cat: idx for idx, cat in enumerate(geodown)}

    DH.saveJson(visdown, '../datas/vis_down/inputs/category.json')
    DH.saveJson(visrep, '../datas/vis_down/inputs/upper_category.json')
    DH.saveJson(geodown, '../datas/geo_down/inputs/category.json')
    DH.saveJson(georep, '../datas/geo_down/inputs/upper_category.json')


def geo_rep2():
    georep_old = DH.loadJson('../datas/geo_down/inputs/backup/upper_category.json')
    geodown_old = DH.loadJson('../datas/geo_down/inputs/backup/category.json')
    georep_new = DH.loadJson('../datas/geo_rep/inputs/category.json')

    geodown_new = set(geodown_old) - set(georep_old)
    geodown_new = geodown_new | set(georep_new)
    geodown_new = sorted(list(geodown_new))
    geodown_new = {cat: idx for idx, cat in enumerate(geodown_new)}

    DH.saveJson(geodown_new, '../datas/geo_down/inputs/category.json')
    DH.saveJson(georep_new, '../datas/geo_down/inputs/upper_category.json')


def get_georep(saved=True, savepath='../datas/geo_rep/inputs/',
               thr=1, hops=2):
    '''
    位置情報について上位クラスを取得
    '''
    from mmm import DataHandler as DH

    # datas = '../datas/geo_rep/inputs/geo/local_df_area16_wocoth_kl5.pickle'
    # datas = './lda_new2.pickle'
    datas = '../datas/bases/local_df_area16_wocoth_new.pickle'
    datas = DH.loadPickle(datas)
    vis_down = DH.loadJson('../datas/vis_down/inputs/category.json')
    # vis_down = [key for key, _ in category.items()]
    vis_reps = DH.loadJson('../datas/vis_down/inputs/upper_category.json')
    # vis_reps = [key for key, _ in vis_reps.items()]
    down_category = list(set(vis_down) - set(vis_reps))

    geo_reps = []
    layer = down_category[:]
    # -------------------------------------------------------------------------
    # n-hop以内にあるrepを取得
    # flgs = []
    # for _ in range(hops):
    #     temp_reps = []
    #     for ccat in layer:
    #         if ccat in flgs:
    #             continue

    #         flgs.append(ccat)
    #         temp_reps.extend(datas['geo_representative'][ccat])

    #     layer = list(set(temp_reps))
    #     geo_reps.extend(layer)

    # geo_reps = list(set(geo_reps) - set(down_category))
    # -------------------------------------------------------------------------
    # n-hop目にあるrepを取得(n-hop目とn-hop未満両方にあるものもrepとして取得)
    for _ in range(hops):
        temp_reps = []
        for ccat in layer:
            temp_reps.extend(datas['geo_representative'][ccat])

        layer = list(set(temp_reps))

    geo_reps = list(set(layer) - set(down_category))
    # -------------------------------------------------------------------------
    geo_reps = [(item, len(datas['geo'][item])) for item in geo_reps]
    geo_reps = [item[0] for item in geo_reps if item[1] >= thr]
    geo_reps.sort()
    geo_reps = {cat: idx for idx, cat in enumerate(geo_reps)}

    if saved:
        DH.saveJson(geo_reps, 'category.json', savepath)

    return geo_reps


def geospatial_df():
    lda = DH.loadPickle('../datas/bases/local_df_area16_wocoth_new.pickle')
    gda = lda[lda['visual'] == 1]

    DH.savePickle(
        gda,
        '../datas/bases/geospatial_df_area16_wocoth.pickle'
    )

    return gda


def limited_category(reps, lda='../datas/bases/local_df_area16_wocoth_new'):
    gr = DH.loadPickle('geo_relationship.pickle', '../datas/bases')
    vis_local = DH.loadJson('category', '../datas/gcn/inputs')
    vis_rep = DH.loadJson('upper_category', '../datas/gcn/inputs')
    local = set(vis_local) - set(vis_rep)

    down = []
    for item in reps:
        down.extend(gr[item])

    down = set(down) & set(local)

    lda = DH.loadPickle(lda)
    down = [item for item in down if len(lda['geo'][item]) > 0]
    down = sorted(list(set(down) | set(reps)))

    return {key: idx for idx, key in enumerate(down)}


def get_geo():
    grd = DH.loadPickle('../datas/bases/geo_rep_df_area16_kl5.pickle')
    geo_rep_dict = grd.to_dict('index')
    lda = DH.loadPickle('../datas/bases/local_df_area16_wocoth_new.pickle')
    local_dict = lda.to_dict('index')
    vis_rep = DH.loadJson('../datas/gcn/inputs/upper_category.json')
    vis_down = DH.loadJson('../datas/gcn/inputs/category.json')
    down_tag = sorted(set(vis_down) - set(vis_rep))

    remove_geo_tag = []
    for tag in geo_rep_dict:
        if len(geo_rep_dict[tag]['down']) == 1:
            remove_geo_tag.append(tag)
    remove_geo_tag = list(set(remove_geo_tag))
    geo_rep = []
    norep_list = []
    for tag in down_tag:
        temp = list(
            set(local_dict[tag]['geo_representative']) - set(remove_geo_tag)
        )
        if temp == []:
            norep_list.append(tag)
        else:
            geo_rep.extend(temp)
    geo_rep = list(set(geo_rep))
    geo_rep2 = []
    for tag in geo_rep:
        geo_rep2.extend(list(
            set(local_dict[tag]['geo_representative']) - set(remove_geo_tag)
        ))

    geo_rep = list(set(geo_rep2) - set(down_tag))
    down_tag = list(set(down_tag) - set(norep_list))
    down_t2f = DH.loadJson('down_tag2file', '../datas/bases/')
    down_t2f = down_t2f['train']
    down_tag = set(down_tag) & set(down_t2f)

    vis_down = sorted(set(down_tag) | set(vis_rep))
    vis_down = {key: idx for idx, key in enumerate(vis_down)}
    geo_rep = sorted(geo_rep)
    geo_rep = {key: idx for idx, key in enumerate(geo_rep)}
    geo_down = sorted(set(down_tag) | set(geo_rep))
    geo_down = {key: idx for idx, key in enumerate(geo_down)}

    DH.saveJson(vis_down, '../datas/vis_down/inputs/category.json')
    DH.saveJson(geo_rep, '../datas/geo_rep/inputs/category.json')
    DH.saveJson(geo_rep, '../datas/geo_down/inputs/upper_category.json')
    DH.saveJson(geo_down, '../datas/geo_down/inputs/category.json')

    return geo_rep, down_tag


if __name__ == "__main__":
    # geospatial_df()
    # vis_tag2file('rep')
    # vis_tag2file('local')

    # vis_sim_dict()
    # aaa = VU.down_mask(
    #     DH.loadJson('upper_category.json', '../datas/gcn/inputs'),
    #     DH.loadJson('category.json', '../datas/gcn/inputs')
    # )
    # bbb = DH.loadPickle('../datas/gcn/inputs/comb_mask/04.pickle')
    # geodata_format()
    get_geo()
    geo_rep2()
    # samedown()

    print('finish.')
