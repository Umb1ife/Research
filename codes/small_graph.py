def small_graph_tags(tag, hops=2):
    from mmm import DataHandler as DH

    lda = '../datas/rr/inputs/geo/local_df_area16_wocoth_kl5.pickle'
    lda = DH.loadPickle(lda)
    rda = '../datas/prepare/rep_df_area16_wocoth.pickle'
    rda = DH.loadPickle(rda)
    grd = '../datas/rr/inputs/geo/geo_rep_df_area16_kl5.pickle'
    grd = DH.loadPickle(grd)
    grd = set(grd.index)

    # -------------------------------------------------------------------------
    # 指定したタグからn-hop以内のdownタグを取得
    layer = rda['down'][tag]
    down_tag, flgs = layer[:], []
    for _ in range(hops - 1):
        temp = []
        for item in layer:
            if item in flgs:
                continue

            flgs.append(item)
            temp.extend(lda['down'][item])

        temp = list(set(temp))
        down_tag.extend(temp)
        layer = temp[:]

    rda = set(rda.index)
    down_tag = sorted(list(set(down_tag) - rda - {tag}))
    down_tag = [
        item for item in down_tag if len(lda['geo_representative'][item]) > 0
    ]

    # -------------------------------------------------------------------------
    # 取得したdownタグからn-hop以内のvisual-repとなるタグを取得
    vis_rep = []
    layer = down_tag[:]
    flgs = []
    for _ in range(hops):
        temp = []
        for item in layer:
            if item in flgs or item in rda:
                continue

            flgs.append(item)
            temp.extend(lda['representative'][item])

        temp = list(set(temp))
        vis_rep.extend(temp)
        layer = temp[:]

    vis_rep = sorted(list((set(vis_rep) - set(down_tag)) & rda))

    # -------------------------------------------------------------------------
    # 取得したdownタグからn-hop以内のgeo-repとなるタグを取得
    geo_rep = []
    layer = down_tag[:]
    flgs = []
    for _ in range(hops):
        temp = []
        for item in layer:
            if item in flgs:
                continue

            flgs.append(item)
            temp.extend(lda['geo_representative'][item])

        temp = list(set(temp))
        geo_rep.extend(temp)
        layer = temp[:]

    geo_rep = sorted(list((set(geo_rep) - set(down_tag)) & grd))

    return vis_rep, geo_rep, down_tag, (len(vis_rep), len(geo_rep), len(down_tag))


if __name__ == "__main__":
    from mmm import DataHandler as DH

    category = DH.loadJson('category.json', '../datas/fine_tuning/inputs')
    category = list(category.keys())
    tag_dict = {key: small_graph_tags(key) for key in category}
    tag_list = [[key] + list(val) for key, val in tag_dict.items()]
    tag_list.sort(key=lambda x: x[4][2])

    tag_pairs = []
    for item1 in tag_list:
        temp = []
        for item2 in tag_list:
            if item1[0] == item2[0]:
                continue

            if max(len(item1[2]), len(item2[2])) == 0:
                sim = 0
            else:
                sim = len(set(item1[2]) & set(item2[2])) \
                    / max(len(item1[2]), len(item2[2]))

            temp.append((sim, item1, item2))

        temp.sort(key=lambda x: -x[0])

        tag_pairs.append(temp)

    print('finish.')
