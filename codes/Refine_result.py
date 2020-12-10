def refining(phase='train', threshold=0.4,
             check_epoch=5, ranges=(0, 12), saved=True):
    '''
    画像をmodelに入力したときの画像表示，正解ラベル，正解ラベルの順位・尤度，全ラベルの尤度を出力

    Arguments:
        phase {'train' or 'validate'} -- 入力が学習用画像か評価用画像かの指定
        threshold {0.1 - 0.9} -- maskのしきい値．予め作成しておくこと
        check_epoch {1 - 20} -- どのepochの段階で評価するかの指定
        ranges {(int, int)} -- 画像集合の何番目から何番目を取ってくるかの指定
    '''
    # ---準備-------------------------------------------------------------------
    import os
    import torch.optim as optim
    from mmm import CustomizedMultiLabelSoftMarginLoss as MyLossFunction
    from mmm import DataHandler as DH
    from mmm import MeanShiftRefiner
    from mmm import MultiLabelGCN
    from tqdm import tqdm

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ranges = ranges if ranges[1] > ranges[0] else (ranges[1], ranges[0])
    ranges = ranges if ranges[1] != ranges[0] else (ranges[0], ranges[0] + 1)
    category = DH.loadJson('category', '../datas/gcn/inputs')
    num_class = len(category)

    # ---学習済みの地理空間概念認識器のload---------------------------------------
    path_top = '../datas/gcn/'

    # maskの読み込み
    mask = DH.loadPickle(
        '{0:0=2}'.format(int(threshold * 10)),
        path_top + 'inputs/comb_mask/'
    )

    # 誤差伝播の重みの読み込み
    bp_weight = DH.loadNpy(
        '{0:0=2}'.format(int(threshold * 10)),
        path_top + 'inputs/backprop_weight/'
    )

    # 学習で用いるデータの設定や読み込み先
    gcn_settings = {
        'class_num': num_class,
        'filepaths': {
            'category': path_top + 'inputs/category.json',
            'upper_category': path_top + 'inputs/upper_category.json',
            'relationship': path_top + 'inputs/relationship.pickle',
            'learned_weight': path_top + 'inputs/learned/200cnn.pth'
        },
        'feature_dimension': 2048
    }

    # modelの設定
    model = MultiLabelGCN(
        class_num=num_class,
        loss_function=MyLossFunction(),
        optimizer=optim.SGD,
        learningrate=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        fix_mask=mask,
        network_setting=gcn_settings,
        multigpu=False,
        backprop_weight=bp_weight
    )

    # 読み込み
    if check_epoch > 1:
        model.loadmodel(
            '{0:0=3}weight'.format(check_epoch),
            path_top + 'outputs/learned/th{0:0=2}'.format(int(threshold * 10))
        )

    # ---MeanShiftRifinerの定義-------------------------------------------------
    path_top = '../datas/geo_rep/'
    local_dict = DH.loadPickle(
        'local_df_area16_wocoth.pickle', path_top + 'inputs'
    )
    local_dict = {item.Index: item.geo for item in local_dict.itertuples()}

    msr = MeanShiftRefiner(local_dict, category)
    msr.save(directory=path_top + 'outputs')

    msr = MeanShiftRefiner()
    msr.load('MSR', directory=path_top + 'outputs')
    photo_location = DH.loadPickle(
        'photo_location_' + phase, path_top + 'inputs'
    )

    # ---画像・正解ラベル・予測ラベルの表示---------------------------------------
    path_top = '../datas/gcn/'
    imlist = DH.loadPickle('imlist', directory=path_top + 'outputs/check/')
    imlist = imlist[ranges[0]:ranges[1]]
    category = [key for key, _ in category.items()]

    results = []
    for idx, (image, label, filename) in enumerate(tqdm(imlist)):
        # 正解ラベル
        label = [lname for flg, lname in zip(label[0], category) if flg == 1]

        # 予測ラベル
        pred = model.predict(image)[0]
        pred = [
            (likelihood, lname) for likelihood, lname in zip(pred, category)
            if likelihood != 0
        ]
        pred = sorted(pred, reverse=True)
        # 予測ラベルのうちtop nまでのもの
        toppred = [(item[1], item[0]) for item in pred]

        # 正解ラベルが予測ではどの順位にあるか
        prank = {tag: (idx, llh) for idx, (llh, tag) in enumerate(pred)}
        prank = [prank[lbl] for lbl in label]

        # 位置情報による絞り込み
        image_loc = photo_location[filename[0]]
        geopred = msr.get_geotags(image_loc)
        rs = {cat: ll for ll, cat in pred}
        geopred = [(rs[gp], gp) for gp in geopred]
        geopred = sorted(geopred, reverse=True)

        # refine後のrecallの確認
        reftags = [item[1] for item in geopred]
        tflist = [lbl in reftags for lbl in label]

        result = {
            # 'filename': filename[0],
            'tag': label,
            # 'tags_rank': prank,
            # 'predict': toppred,
            'refined': geopred,
            'correct_or_not': tflist,
            'correct_answer_rate': sum(tflist) / len(tflist)
        }

        results.append(result)

    if saved:
        DH.savePickle(results, 'RR', '../datas/geo_rep/outputs/')

    return results


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    r = refining(ranges=(0, 100))
    # -------------------------------------------------------------------------
    print('finish.')
