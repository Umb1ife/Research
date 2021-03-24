import os
import sys

sys.path.append(os.path.join('..', 'codes'))


def confusion_all_matrix(epoch=20, saved=True,
                         weight_path='../datas/vis_down/outputs/learned/',
                         outputs_path='../datas/vis_down/outputs/check/'):
    '''
    正例・unknown・負例についてconfusion_matrixを作成
    '''
    # -------------------------------------------------------------------------
    # 準備
    import numpy as np
    import os
    import torch
    from mmm import DataHandler as DH
    from mmm import DatasetFlickr
    from mmm import VisGCN
    from mmm import VisUtils as VU
    from torchvision import transforms
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # データの読み込み先
    input_path = '../datas/vis_down/inputs/'
    category = DH.loadJson('category.json', input_path)
    rep_category = DH.loadJson('upper_category.json', input_path)

    vis_down_train = VU.down_anno(category, rep_category, 'train')
    vis_down_validate = VU.down_anno(category, rep_category, 'validate')
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=(1.0, 1.0), ratio=(1.0, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    kwargs_DF = {
        'train': {
            'category': category,
            'annotations': vis_down_train,
            'transform': transform,
            'image_path': input_path + 'images/train/'
        },
        'validate': {
            'category': category,
            'annotations': vis_down_validate,
            'transform': transform,
            'image_path': input_path + 'images/validate/'
        }
    }

    train_dataset = DatasetFlickr(**kwargs_DF['train'])
    val_dataset = DatasetFlickr(**kwargs_DF['validate'])
    num_class = train_dataset.num_category()

    # maskの読み込み
    mask = VU.down_mask(rep_category, category, sim_thr=0.4, saved=False)

    # 学習で用いるデータの設定や読み込み先
    gcn_settings = {
        'category': category,
        'rep_category': rep_category,
        'relationship': DH.loadPickle('relationship.pickle', input_path),
        'rep_weight': torch.load(input_path + 'rep_weight.pth'),
        'feature_dimension': 2048
    }

    # modelの設定
    model = VisGCN(
        class_num=num_class,
        weight_decay=1e-4,
        fix_mask=mask,
        network_setting=gcn_settings,
    )
    model.loadmodel('{0:0=3}weight'.format(epoch), weight_path)

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

    def count_result(dataset):
        from mmm import MakeBPWeight

        loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=1,
            num_workers=4
        )
        bp_weight = MakeBPWeight(dataset, len(category), mask)

        allnum = 0
        counts = np.zeros((len(category), 8))
        for locate, label, _ in tqdm(loader):
            allnum += 1
            fix_mask = _update_backprop_weight(label, mask)
            predicts = model.predict(locate, labeling=True)
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

        for idx, (zero, one) in enumerate(bp_weight):
            counts[idx][3] = one
            counts[idx][5] = allnum - one - zero
            counts[idx][7] = zero

            if counts[idx][2] + counts[idx][6] != 0:
                counts[idx][0] = counts[idx][2] / (counts[idx][2] + counts[idx][6])
            if counts[idx][3] != 0:
                counts[idx][1] = counts[idx][2] / counts[idx][3]

        return counts

    train_counts = count_result(train_dataset)
    validate_counts = count_result(val_dataset)

    if saved:
        DH.saveNpy(
            np.array(train_counts),
            'cm_train_{0:0=3}'.format(epoch),
            outputs_path
        )
        DH.saveNpy(
            np.array(validate_counts),
            'cm_validate_{0:0=3}'.format(epoch),
            outputs_path
        )

    return np.array(train_counts), np.array(validate_counts)


def rep_confmat(saved=False, output_path='../datas/vis_down/outputs/check/'):
    '''
    トレーニング前後での精度の変化、各クラス、各クラスの上位クラス
    を一覧にしたリストの作成
    '''
    import numpy as np
    from mmm import DataHandler as DH
    from tqdm import tqdm

    epoch00 = DH.loadNpy(
        'cm_train_000.npy',
        '../datas/vis_down/outputs/check/learned_lmask/'
    )
    epoch20 = DH.loadNpy(
        'cm_train_020.npy',
        '../datas/vis_down/outputs/check/learned_lmask/'
    )

    category = DH.loadJson('../datas/vis_down/inputs/category.json')
    rep_category = DH.loadJson('../datas/vis_down/inputs/upper_category.json')
    ldf = DH.loadPickle('../datas/bases/local_df_area16_wocoth_new.pickle')
    # -------------------------------------------------------------------------
    # 0: label, 1: rep
    # 2 ~ 9: confusion_all_matrix of epoch 0
    # 10 ~ 17: confusion_all matrix of epoch 20

    compare_list = []
    for idx, cat in tqdm(enumerate(category)):
        if cat in rep_category:
            continue

        row = [cat, ldf.loc[cat]['representative']]
        row.extend(epoch00[idx])
        row.extend(epoch20[idx])

        compare_list.append(row)

    compare_list = np.array(compare_list, dtype=object)

    if saved:
        DH.saveNpy(compare_list, 'compare_pr', output_path)

    return compare_list


def hist_change(phase='train', barnum=3, binnum=5, saved=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from mmm import DataHandler as DH

    plt.rcParams['font.family'] = 'IPAexGothic'
    # -------------------------------------------------------------------------
    data = DH.loadNpy('../datas/vis_down/outputs/check/compare_pr.npy')
    before = [[] for _ in range(barnum)]
    after = [[] for _ in range(barnum)]
    ml = 0
    for item in data:
        idx = len(item[1]) - 1
        ml = max(idx, ml)
        idx = min(idx, barnum - 1)
        before[idx].append(item[3])
        after[idx].append(item[11])

    before = [item for item in before if item]
    after = [item for item in after if item]
    barnum = len(before)
    before_heights = np.zeros((barnum, binnum))
    after_heights = np.zeros((barnum, binnum))
    thresholds = np.linspace(0.0, 1.0, binnum + 1)

    for idx, item in enumerate(before):
        bin_heights = np.histogram(item, bins=binnum, range=(0.0, 1.0))[0]
        before_heights[idx] = bin_heights / sum(bin_heights)

    for idx, item in enumerate(after):
        bin_heights = np.histogram(item, bins=binnum, range=(0.0, 1.0))[0]
        after_heights[idx] = bin_heights / sum(bin_heights)

    # -------------------------------------------------------------------------
    x = np.arange(binnum)
    width = 0.8 / barnum
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # -------------------------------------------------------------------------
    ax1.bar(x + 0.5, np.ones(binnum), 0.8, alpha=0.15)
    for idx, bh in enumerate(before_heights[:-1]):
        ax1.bar(
            x + 0.1 + width * (idx + 0.5),
            bh, width,
            label='{0}個'.format(idx + 1)
        )

    ll = '{0}個'.format(barnum)
    ll = ll + '以上' if ml + 1 > barnum else ll
    ax1.bar(
        x + 0.1 + width * (barnum - 0.5),
        before_heights[-1], width,
        label=ll
    )

    labels = ['{0}'.format(int(item * 100)) for item in thresholds]

    # ax.set_title(fig_title)
    # ax1.set_xlabel('学習前の再現率(%)')
    # ax1.set_ylabel('割合')
    ax1.set_xticks(np.arange(binnum + 1))
    ax1.set_xticklabels(labels)
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(['0 %', '25 %', '50 %', '75 %', '100 %'])
    ax1.set_aspect(1.5)
    ax1.legend(title='視覚代表概念の数')

    # -------------------------------------------------------------------------
    ax2.bar(x + 0.5, np.ones(binnum), 0.8, alpha=0.15)
    for idx, bh in enumerate(after_heights[:-1]):
        ax2.bar(
            x + 0.1 + width * (idx + 0.5),
            bh, width,
            label='{0}個'.format(idx + 1)
        )

    ll = '{0}個'.format(barnum)
    ll = ll + '以上' if ml + 1 > barnum else ll
    ax2.bar(
        x + 0.1 + width * (barnum - 0.5),
        after_heights[-1], width,
        label=ll
    )

    labels = ['{0}'.format(int(item * 100)) for item in thresholds]

    # ax.set_title(fig_title)
    ax2.set_xlabel('学習前後での再現率(%)\n(上：学習前, 下：学習後)')
    # ax2.set_ylabel('割合')
    ax2.set_xticks(np.arange(binnum + 1))
    ax2.set_xticklabels(labels)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(['0 %', '25 %', '50 %', '75 %', '100 %'])
    ax2.set_aspect(1.5)
    # ax2.legend(title='視覚代表概念の数', loc='upper left')

    # -------------------------------------------------------------------------
    if saved:
        directory = '../datas/vis_down/outputs/check/images'
        os.makedirs(directory, exist_ok=True)
        fig.savefig(
            '{0}/{1}.png'.format(directory, phase),
            bbox_inches="tight",
            pad_inches=0.1
        )

    fig.show()


def predict_sample(epoch=20, phase='train', saved=True, num=3, thr=0.5,
                   max_falsepositive=5,
                   weight_path='../datas/vis_down/outputs/learned_lr1_bp20/',
                   outputs_path='../datas/vis_down/outputs/check/sample/'):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import torch
    import torch.backends.cudnn as cudnn
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
    from mmm import DataHandler as DH
    from mmm import DatasetFlickr
    from mmm import VisGCN
    from mmm import VisUtils as VU
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # データの読み込み先
    input_path = '../datas/vis_down/inputs/'
    category = DH.loadJson('category.json', input_path)
    rep_category = DH.loadJson('upper_category.json', input_path)
    num_class = len(category)

    vis_down_train = VU.down_anno(category, rep_category, 'train')
    vis_down_validate = VU.down_anno(category, rep_category, 'validate')
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=(1.0, 1.0), ratio=(1.0, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    kwargs_DF = {
        'train': {
            'category': category,
            'annotations': vis_down_train,
            'transform': transform,
            'image_path': input_path + 'images/train/'
        },
        'validate': {
            'category': category,
            'annotations': vis_down_validate,
            'transform': transform,
            'image_path': input_path + 'images/validate/'
        }
    }

    dataset = DatasetFlickr(**kwargs_DF[phase])
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=4
    )

    if torch.cuda.is_available():
        loader.pin_memory = True
        cudnn.benchmark = True

    # 学習で用いるデータの設定や読み込み先
    gcn_settings = {
        'category': category,
        'rep_category': rep_category,
        'relationship': DH.loadPickle('relationship.pickle', input_path),
        'rep_weight': torch.load(input_path + 'rep_weight.pth'),
        'feature_dimension': 2048
    }

    # modelの設定
    model = VisGCN(
        class_num=num_class,
        weight_decay=1e-4,
        network_setting=gcn_settings,
    )
    model.loadmodel('{0:0=3}weight'.format(epoch), weight_path)

    # -------------------------------------------------------------------------
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

    nacategory = np.array(list(category.keys()))
    num = max(min(num, len(dataset)), 0)
    bar = tqdm(total=num)
    for idx, (images, labels, filename) in enumerate(loader):
        if idx >= num:
            break

        true_label = nacategory[labels[0] > 0]
        prd = model.predict(images)[0]
        predict_label = nacategory[prd >= thr]

        image_s = Image.open(
            os.path.join(kwargs_DF[phase]['image_path'], filename[0])
        ).convert('RGB')
        image_s = expand2square(image_s)
        xl = ''
        for tl in true_label:
            xl += tl + ': {0:.2f}%\n'.format(prd[category[tl]] * 100)

        pls = [
            (pl, prd[category[pl]])
            for pl in predict_label if pl not in true_label
        ]
        pls = sorted(pls, key=lambda x: -x[1])
        fpl = ''
        for pl, ll in pls[:max_falsepositive]:
            fpl += pl + ': {0:.2f}%\n'.format(ll * 100)

        if len(pls) > max_falsepositive:
            fpl += '~\n'

        ax = plt.subplot(111)
        ax.imshow(image_s)
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False
        )
        truelabels = TextArea(xl[:-1], textprops=dict(color="b", size=14))
        falsepositive = TextArea(fpl[:-1], textprops=dict(color="r", size=14))
        chl = [truelabels, falsepositive] if fpl != '' else [truelabels]
        xbox = VPacker(children=chl, align="left", pad=0, sep=5)

        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0)
        ax.add_artist(anchored_xbox)

        plt.tight_layout()
        plt.savefig(
            '../datas/vis_down/outputs/check/samples/' + filename[0],
            bbox_inches="tight"
        )
        plt.show()
        plt.clf()

        bar.update(1)
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    # predict_sample(saved=True, num=1000)
    # rep_confmat(saved=True)
    # hist_change(saved=True)
    confusion_all_matrix(
        epoch=20,
        weight_path='../datas/vis_down/outputs/learned/',
        outputs_path='../datas/vis_down/outputs/check/learned/'
    )
    confusion_all_matrix(
        epoch=0,
        weight_path='../datas/vis_down/outputs/learned/',
        outputs_path='../datas/vis_down/outputs/check/learned/'
    )

    print('finish.')
