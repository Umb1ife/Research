import numpy as np
import os
import torch
from geodown_training import limited_category
from mmm import DataHandler as DH
from mmm import GeotagGCN
from mmm import GeoUtils as GU
from mmm import MultiLabelGCN
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def get_data(anno, phase='train'):
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose(
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
    )

    filename = anno['file_name']
    image_path = '../datas/gcn/inputs/images/{0}/'.format(phase)
    image = Image.open(
        os.path.join(image_path, filename)
    ).convert('RGB')
    image = tfm(image).unsqueeze(0)

    return image


def recognize():
    # -------------------------------------------------------------------------
    vis_rep = DH.loadJson('upper_category.json', '../datas/gcn/inputs')
    vis_down = DH.loadJson('category.json', '../datas/gcn/inputs')
    geo_rep = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    geo_down = limited_category(geo_rep)

    category = sorted(list(set(geo_down) - set(geo_rep) - set(vis_rep)))
    category = {cat: idx for idx, cat in enumerate(category)}
    vdlist = np.array(list(vis_down))
    gdlist = np.array(list(geo_down))
    # -------------------------------------------------------------------------
    vispath = '../datas/gcn/inputs/'
    gcn_settings = {
        'num_class': len(vis_down),
        'filepaths': {
            'category': vispath + 'category.json',
            'upper_category': vispath + 'upper_category.json',
            'relationship': vispath + 'relationship.pickle',
            # 'learned_weight': vispath + 'learned/200cnn.pth'
            'learned_weight': vispath + '200cnn.pth'
        },
        'feature_dimension': 2048
    }
    vis_model = MultiLabelGCN(
        class_num=len(vis_down),
        # loss_function=MyLossFunction(),
        # optimizer=optim.SGD,
        # learningrate=args.learning_rate,
        # momentum=0.9,
        weight_decay=1e-4,
        # fix_mask=mask,
        network_setting=gcn_settings,
        # multigpu=True if len(args.device_ids.split(',')) > 1 else False,
        # backprop_weight=bp_weight
    )
    vis_model.loadmodel('../datas/gcn/outputs/learned_backup/020weight.pth')
    # -------------------------------------------------------------------------
    gcn_settings = {
        'category': geo_down,
        'rep_category': geo_rep,
        'filepaths': {
            'relationship': '../datas/bases/geo_relationship.pickle',
            # 'learned_weight': '../datas/geo_rep/outputs/learned/200weight.pth'
            'learned_weight': '../datas/geo_rep/outputs/learned_nobp_zeroag10_none/200weight.pth'
        },
        'base_weight_path': '../datas/geo_base/outputs/learned/200weight.pth',
        'BR_settings': {'fineness': (20, 20)},
    }

    # modelの設定
    geo_model = GeotagGCN(
        class_num=len(geo_down),
        # loss_function=MyLossFunction(reduction='none'),
        # optimizer=optim.SGD,
        # learningrate=args.learning_rate,
        # momentum=0.9,
        weight_decay=1e-4,
        # fix_mask=mask,
        network_setting=gcn_settings,
        # multigpu=True if len(args.device_ids.split(',')) > 1 else False,
        # backprop_weight=bp_weight
    )
    geo_model.loadmodel('../datas/geo_down/outputs/learned_base_bp2/020weight.pth')
    # -------------------------------------------------------------------------
    pld2017 = DH.loadPickle('photo_loc_dict_2017.pickle',
                            '../datas/backup/prepare/inputs')
    pld2016 = DH.loadPickle('photo_loc_dict_2016.pickle',
                            '../datas/backup/prepare/inputs')
    ta = DH.loadJson('train_anno.json', '../datas/gcn/inputs')
    va = DH.loadJson('validate_anno.json', '../datas/gcn/inputs')
    # -------------------------------------------------------------------------
    ans_pre_pairs_train = []
    for item in tqdm(ta):
        image_data = get_data(item, 'train')
        locate = pld2017[int(item['file_name'][:-4])]
        true_label = list(set(vdlist[item['labels']]) & set(category))
        if not true_label:
            continue

        vout = vis_model.predict(image_data, normalized=True, labeling=True)
        gout = geo_model.predict(torch.Tensor(locate), labeling=True)

        vcat = vdlist[np.where(vout > 0)[1]]
        gcat = gdlist[np.where(gout > 0)[1]]

        predict = set(vcat) & set(gcat) & set(category)
        ans_pre_pairs_train.append((
            true_label,
            predict,
            set(vcat) & set(category),
            set(gcat) & set(category),
            item['file_name'],
            locate
        ))

    DH.savePickle(ans_pre_pairs_train, 'app_train', '../datas/last/')

    ans_pre_pairs_validate = []
    for item in tqdm(va):
        image_data = get_data(item, 'validate')
        locate = pld2016[int(item['file_name'][:-4])]
        true_label = list(set(vdlist[item['labels']]) & set(category))
        if not true_label:
            continue

        vout = vis_model.predict(image_data, normalized=True, labeling=True)
        gout = geo_model.predict(torch.Tensor(locate), labeling=True)

        vcat = vdlist[np.where(vout > 0)[1]]
        gcat = gdlist[np.where(gout > 0)[1]]

        predict = set(vcat) & set(gcat) & set(category)
        ans_pre_pairs_validate.append((
            true_label,
            predict,
            set(vcat) & set(category),
            set(gcat) & set(category),
            item['file_name'],
            locate
        ))

    DH.savePickle(ans_pre_pairs_train, 'app_validate', '../datas/last/')

    return


def iweight(labels, mask, category):
    weight = np.zeros(mask.shape[0])
    self_idx = []
    for cat in labels:
        cidx = category[cat]
        # weight = weight | mask[cidx]
        weight = (weight > 0) | (mask[cidx] > 0)
        self_idx.append(cidx)

    weight[cidx] = 0
    weight = np.ones(mask.shape[0]) - weight

    return weight


def test(phase='train'):
    app = DH.loadPickle('app_{0}.pickle'.format(phase))

    vis_rep = DH.loadJson('upper_category.json', '../datas/gcn/inputs')
    vis_down = DH.loadJson('category.json', '../datas/gcn/inputs')
    geo_rep = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    geo_down = limited_category(geo_rep)

    catlist = np.array(sorted(list(
        set(geo_down) - set(geo_rep) - set(vis_rep)
    )))
    category = {cat: idx for idx, cat in enumerate(catlist)}
    # -------------------------------------------------------------------------
    cnt_rc = np.zeros((len(category), 2))
    cnt_pr = np.zeros((len(category), 2))
    seps = []
    lm = lastmask(category)
    for apval in tqdm(app):
        tl, prd, vp, gp, fn, lc = apval
        # tl, prd = apval
        iw = iweight(tl, lm, category)
        afmask = set(catlist[iw > 0])
        prd = afmask & prd

        seps.append((
            tl,
            prd,
            vp & afmask,
            gp & afmask,
            len(set(tl) & prd) / max(len(tl), len(prd)),
            lc,
            vp,
            gp,
            vp & gp,
            fn
        ))

        for item in tl:
            cnt_rc[category[item]][1] += 1
            if item in prd:
                cnt_rc[category[item]][0] += 1

        for item in prd:
            cnt_pr[category[item]][1] += 1
            if item in tl:
                cnt_pr[category[item]][0] += 1

    recall = cnt_rc.sum(axis=0)
    recall = recall[0] / recall[1]
    precision = cnt_pr.sum(axis=0)
    precision = precision[0] / precision[1]

    return recall, precision


def lastmask(category=None):
    vis_down = DH.loadJson('category.json', '../datas/gcn/inputs')
    geo_rep = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    geo_down = limited_category(geo_rep)
    # vis_rep = DH.loadJson('upper_category.json', '../datas/gcn/inputs')
    # category = sorted(list(set(geo_down) - set(geo_rep) - set(vis_rep)))
    # category = {cat: idx for idx, cat in enumerate(category)}

    vdlist = np.array(list(vis_down))
    gdlist = np.array(list(geo_down))

    vismask = DH.loadPickle('04.pickle', '../datas/gcn/inputs/comb_mask')
    geomask = GU.down_mask(geo_rep, geo_down, saved=False)

    lmask = np.zeros((len(category), len(category)))
    for cat, idx in category.items():
        vissim = vdlist[vismask[vis_down[cat]] > 0]
        geosim = gdlist[geomask[geo_down[cat]] > 0]

        lsim = set(vissim) & set(geosim)
        lsim_idx = [category[item] for item in lsim]
        temp = np.zeros(len(category))
        temp[lsim_idx] = 1

        lmask[idx] = temp

    np.fill_diagonal(lmask, 0)

    return lmask


if __name__ == "__main__":
    # lastmask()
    test()
    recognize()
