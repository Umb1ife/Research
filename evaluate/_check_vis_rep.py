import os
import sys

sys.path.append(os.path.join('..', 'codes'))


def confusion_all_matrix(epoch=200, saved=True,
                         weight_path='../datas/geo_down/outputs/learned/',
                         outputs_path='../datas/geo_down/outputs/check/'):
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
    from mmm import FinetuneModel
    from mmm import VisUtils as VU
    from torchvision import transforms
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # データの読み込み先
    input_path = '../datas/vis_rep/inputs/'
    category = DH.loadJson('category.json', input_path)

    vis_rep_train = VU.rep_anno(category, phase='train')
    vis_rep_validate = VU.rep_anno(category, phase='validate')
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
            'annotations': vis_rep_train,
            'transform': transform,
            'image_path': input_path + 'images/train/'
        },
        'validate': {
            'category': category,
            'annotations': vis_rep_validate,
            'transform': transform,
            'image_path': input_path + 'images/validate/'
        }
    }

    train_dataset = DatasetFlickr(**kwargs_DF['train'])
    val_dataset = DatasetFlickr(**kwargs_DF['validate'])
    num_class = train_dataset.num_category()

    # maskの読み込み
    mask = VU.rep_mask(category)

    # modelの設定
    model = FinetuneModel(
        class_num=num_class,
        momentum=0.9,
        fix_mask=mask,
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


def predict_sample(epoch=200, phase='train', saved=True, num=3, thr=0.5,
                   weight_path='../datas/vis_rep/outputs/learned/',
                   outputs_path='../datas/vis_rep/outputs/check/sample/'):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import torch
    import torch.backends.cudnn as cudnn
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
    from mmm import DataHandler as DH
    from mmm import DatasetFlickr
    from mmm import FinetuneModel
    from mmm import VisUtils as VU
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # データの読み込み先
    input_path = '../datas/vis_rep/inputs/'
    category = DH.loadJson('category.json', input_path)
    num_class = len(category)

    vis_rep_train = VU.rep_anno(category, phase='train')
    vis_rep_validate = VU.rep_anno(category, phase='validate')
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
            'annotations': vis_rep_train,
            'transform': transform,
            'image_path': input_path + 'images/train/'
        },
        'validate': {
            'category': category,
            'annotations': vis_rep_validate,
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

    # maskの読み込み
    mask = VU.rep_mask(category)

    # modelの設定
    model = FinetuneModel(
        class_num=num_class,
        momentum=0.9,
        fix_mask=mask,
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
        for pl, ll in pls:
            fpl += pl + ': {0:.2f}%\n'.format(ll * 100)

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
            '../datas/vis_rep/outputs/check/samples/' + filename[0],
            bbox_inches="tight"
        )
        plt.show()
        plt.clf()

        bar.update(1)
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    # predict_sample(saved=False, num=3)
    predict_sample(saved=True, num=100000)
    # confusion_all_matrix(
    #     epoch=200,
    #     weight_path='../datas/vis_rep/outputs/learned/',
    #     outputs_path='../datas/vis_rep/outputs/check/learned/'
    # )
    # confusion_all_matrix(
    #     epoch=0,
    #     weight_path='../datas/vis_rep/outputs/learned/',
    #     outputs_path='../datas/vis_rep/outputs/check/learned/'
    # )

    print('finish.')
