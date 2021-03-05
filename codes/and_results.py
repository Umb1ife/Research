import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from mmm import DataHandler as DH
from mmm import GeotagGCN
from mmm import GeoUtils as GU
from mmm import VisGCN
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def expand2square(pil_img, bg_color='white'):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), bg_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), bg_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


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
    image_path = '../datas/vis_down/inputs/images/{0}/'.format(phase)
    image = Image.open(
        os.path.join(image_path, filename)
    ).convert('RGB')
    image = tfm(image).unsqueeze(0)

    return image


def iweight(labels, mask, category):
    weight = np.zeros(mask.shape[0])
    self_idx = []
    for cat in labels:
        cidx = category[cat]
        weight = (weight > 0) | (mask[cidx] > 0)
        self_idx.append(cidx)

    weight = np.ones(mask.shape[0]) - weight
    weight[self_idx] = 1

    return weight


def lastmask(category=None):
    vis_down = DH.loadJson('category.json', '../datas/vis_down/inputs')
    geo_down = DH.loadJson('category.json', '../datas/geo_down/inputs')

    vdlist = np.array(list(vis_down))
    gdlist = np.array(list(geo_down))

    vismask = DH.loadPickle('04.pickle', '../datas/vis_down/inputs')
    geomask = DH.loadPickle('mask_5.pickle', '../datas/geo_down/inputs')

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


def recognize(saved=True):
    # -------------------------------------------------------------------------
    vis_rep = DH.loadJson('category.json', '../datas/vis_rep/inputs')
    vis_down = DH.loadJson('category.json', '../datas/vis_down/inputs')
    geo_rep = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    geo_down = DH.loadJson('category.json', '../datas/geo_down/inputs')

    category = set(vis_down) - set(vis_rep)
    category = sorted(list(category))
    ctlist = np.array(category)
    category = {cat: idx for idx, cat in enumerate(category)}
    catset = set(category)
    vdlist = np.array(list(vis_down))
    gdlist = np.array(list(geo_down))

    # -------------------------------------------------------------------------
    vispath = '../datas/vis_down/inputs/'
    visgcn_settings = {
        'category': vis_down,
        'rep_category': vis_rep,
        'relationship': DH.loadPickle('relationship.pickle', vispath),
        'rep_weight': torch.load(vispath + 'rep_weight.pth'),
        'feature_dimension': 2048
    }
    vis_model = VisGCN(
        class_num=len(vis_down),
        weight_decay=1e-4,
        network_setting=visgcn_settings,
    )
    vis_model.loadmodel(
        '020weight.pth', '../datas/vis_down/outputs/learned_lmask/'
    )
    # -------------------------------------------------------------------------
    geogcn_settings = {
        'category': geo_down,
        'rep_category': geo_rep,
        'relationship': DH.loadPickle('../datas/bases/geo_relationship.pickle'),
        'rep_weight': torch.load('../datas/geo_rep/outputs/learned/200weight.pth'),
        'base_weight': torch.load(
            '../datas/geo_base/outputs/learned_50x25/400weight.pth'
        ),
        'BR_settings': {'fineness': (50, 25)}
    }

    # modelの設定
    geo_model = GeotagGCN(
        class_num=len(geo_down),
        weight_decay=1e-4,
        network_setting=geogcn_settings,
    )
    geo_model.loadmodel(
        '020weight.pth', '../datas/geo_down/outputs/learned_rep32_bp_lmask'
    )
    # -------------------------------------------------------------------------
    pld2017 = DH.loadPickle('photo_loc_dict_2017.pickle',
                            '../datas/backup/prepare/inputs')
    pld2016 = DH.loadPickle('photo_loc_dict_2016.pickle',
                            '../datas/backup/prepare/inputs')
    ta = DH.loadJson('train_anno.json', '../datas/vis_down/inputs')
    va = DH.loadJson('validate_anno.json', '../datas/vis_down/inputs')
    gdt = GU.down_dataset(geo_rep, geo_down, 'train')
    gdv = GU.down_dataset(geo_rep, geo_down, 'validate')
    gdt = {tuple(item['locate']): item['labels'] for item in gdt}
    gdv = {tuple(item['locate']): item['labels'] for item in gdv}
    # -------------------------------------------------------------------------
    data_train = []
    for item in tqdm(ta):
        image_data = get_data(item, 'train')
        locate = pld2017[int(item['file_name'][:-4])]
        # if locate not in gdt:
        #     continue

        # loc_label = set(gdlist[gdt[locate]]) & catset
        true_label = set(vdlist[item['labels']]) & catset
        # if loc_label > true_label or not true_label:
        #     continue

        vout = vis_model.predict(image_data, normalized=True)
        gout = geo_model.predict(torch.Tensor(locate))
        vout = vout[0][np.isin(vdlist, ctlist)]
        gout = gout[0][np.isin(gdlist, ctlist)]

        data_train.append((
            sorted(list(true_label)),
            vout,
            gout,
            item['file_name'],
            locate
        ))

    if saved:
        DH.savePickle(data_train, 'recog_train_all', '../datas/last/')

    data_validate = []
    for item in tqdm(va):
        image_data = get_data(item, 'validate')
        locate = pld2016[int(item['file_name'][:-4])]
        # if locate not in gdv:
        #     continue

        # loc_label = set(gdlist[gdv[locate]]) & catset
        true_label = set(vdlist[item['labels']]) & set(category)
        # if loc_label > true_label or not true_label:
        #     continue

        vout = vis_model.predict(image_data, normalized=True, labeling=True)
        gout = geo_model.predict(torch.Tensor(locate), labeling=True)
        vout = vout[0][np.isin(vdlist, ctlist)]
        gout = gout[0][np.isin(gdlist, ctlist)]

        data_validate.append((
            sorted(list(true_label)),
            vout,
            gout,
            item['file_name'],
            locate
        ))

    if saved:
        DH.savePickle(data_validate, 'recog_validate_all', '../datas/last/')

    return data_train, data_validate


def lastresult(phase='train', before_labeling=False, thr=0.5, blthr=0.5):
    rcgs = DH.loadPickle('../datas/last/recog_{0}_all.pickle'.format(phase))
    geo_rep = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    geo_down = DH.loadJson('category.json', '../datas/geo_down/inputs')

    catset = set(geo_down) - set(geo_rep)
    catlist = np.array(sorted(catset))
    category = {cat: idx for idx, cat in enumerate(catlist)}
    lm = lastmask(category)
    # -------------------------------------------------------------------------
    cnt_rc = np.zeros((len(category), 2))
    cnt_pr = np.zeros((len(category), 2))
    conf_mat = np.zeros((2, 2))
    results = []
    for rcg in tqdm(rcgs):
        true_label, vprd, gprd, filename, locate = rcg
        iw = iweight(true_label, lm, category)
        if before_labeling:
            vprd[vprd >= blthr] = 1
            vprd[vprd < blthr] = 0
            gprd[gprd >= blthr] = 1
            gprd[gprd < blthr] = 0

        lprd = vprd * gprd * iw
        # lprd = vprd * gprd
        pidx = lprd >= thr
        prd = catlist[pidx]

        results.append((
            true_label,
            prd,
            len(set(true_label) & set(prd)) / max(len(true_label), len(prd)),
            lprd,
            vprd,
            gprd,
            filename,
            locate
        ))

        for item in true_label:
            cnt_rc[category[item]][1] += 1
            if item in prd:
                cnt_rc[category[item]][0] += 1
                conf_mat[1][1] += 1
            else:
                conf_mat[1][0] += 1

        for item in prd:
            cnt_pr[category[item]][1] += 1
            if item in true_label:
                cnt_pr[category[item]][0] += 1
            else:
                conf_mat[0][1] += 1

        conf_mat[0][0] += len(catset - set(prd) - set(true_label))

    recall = cnt_rc.sum(axis=0)
    recall = recall[0] / recall[1]
    precision = cnt_pr.sum(axis=0)
    precision = precision[0] / precision[1]

    results = sorted(results, key=lambda x: -x[2])
    fkey_rdict = {item[6]: item[:3] for item in results}

    return results
    # return fkey_rdict


def coverage(phase='train'):
    rcgs = DH.loadPickle('../datas/last/recog_{0}_all.pickle'.format(phase))
    geo_rep = DH.loadJson('category.json', '../datas/geo_rep/inputs')
    geo_down = DH.loadJson('category.json', '../datas/geo_down/inputs')

    catlist = np.array(sorted(list(set(geo_down) - set(geo_rep))))
    category = {cat: idx for idx, cat in enumerate(catlist)}
    lm = lastmask(category)
    # -------------------------------------------------------------------------
    lank_list = []
    for rcg in tqdm(rcgs):
        true_label, vprd, gprd, filename, locate = rcg
        iw = iweight(true_label, lm, category)
        lprd = vprd * gprd * iw
        sprd = np.array(sorted(list(zip(lprd, catlist)), key=lambda x: -x[0]))
        tlset = set(true_label)
        for lank, (_, tag) in enumerate(sprd):
            tlset.discard(tag)
            if not tlset:
                lank_list.append(lank + 1)
                break

    return lank_list


def predict_samples(phase='train', num_images=10000, max_falsepositive=5):
    results = lastresult(phase=phase, before_labeling=True)
    dgt = len(str(len(results)))

    for idx, item in enumerate(tqdm(results[:num_images])):
        filename = item[6]
        true_label, predict_label = item[0], item[1]
        ipath = '../datas/vis_down/inputs/images/{0}'.format(phase)
        image_s = Image.open(os.path.join(ipath, filename)).convert('RGB')
        image_s = expand2square(image_s)

        xl, fpl = '', ''
        for tl in true_label:
            if tl in predict_label:
                xl += tl + '\n'

        fcnt = 1
        for pl in predict_label:
            if fcnt > max_falsepositive:
                fpl += '~\n'
                break

            if pl not in true_label:
                fpl += pl + '\n'
                fcnt += 1

        xl = xl[:-1]
        fpl = fpl[:-1]

        ax = plt.subplot(111)
        ax.imshow(image_s)
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False
        )
        truelabels = TextArea(xl, textprops=dict(color="b", size=14))
        falsepositive = TextArea(fpl, textprops=dict(color="r", size=14))
        chl = [truelabels] if xl else []
        chl = chl + [falsepositive] if fpl else chl
        xbox = VPacker(children=chl, align="left", pad=0, sep=5)

        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0)
        ax.add_artist(anchored_xbox)

        plt.tight_layout()
        savename = '{0:0={1}}.jpg'.format(idx, dgt)
        plt.savefig(
            '../datas/last/pimages/' + savename,
            bbox_inches="tight"
        )
        plt.show()
        plt.clf()


def predict_sample(filename, fkey_rdict, phase='train', max_falsepositive=5):
    def expand2square(pil_img, bg_color='white'):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), bg_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), bg_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    # -------------------------------------------------------------------------
    ipath = '../datas/vis_down/inputs/images/{0}'.format(phase)
    image_s = Image.open(os.path.join(ipath, filename)).convert('RGB')
    image_s = expand2square(image_s)
    true_label, predict_label, prc = fkey_rdict[filename]

    xl, fpl = '', ''
    for pl in predict_label:
        if pl in true_label:
            xl += pl + '\n'
        else:
            fpl += pl + '\n'

    xl = xl[:-1]
    fpl = fpl[:-1]

    ax = plt.subplot(111)
    ax.imshow(image_s)
    ax.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False
    )
    truelabels = TextArea(xl, textprops=dict(color="b", size=14))
    falsepositive = TextArea(fpl, textprops=dict(color="r", size=14))
    chl = [truelabels] if xl else []
    chl = chl + [falsepositive] if fpl else chl
    xbox = VPacker(children=chl, align="left", pad=0, sep=5)

    anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0)
    ax.add_artist(anchored_xbox)

    plt.tight_layout()
    plt.savefig(
        '../datas/vis_down/outputs/check/samples/' + filename,
        bbox_inches="tight"
    )
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # predict_samples(num_images=10000)
    # lastmask()
    # recognize()
    # coverage()
    lastresult(before_labeling=True, thr=0.5)

    print('finish.')
