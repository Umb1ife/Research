import numpy as np
from .datahandler import DataHandler as DH
from tqdm import tqdm


class VisUtils:
    @staticmethod
    def rep_anno(category, phase='train', base_path='../datas/bases/'):
        print('preparing dataset: {0} ...'.format(phase))
        t2f = DH.loadJson('rep_tag2file', base_path)
        t2f = t2f[phase]
        photo2tag = {}
        for cat in category:
            for pf in t2f[cat]['filename']:
                if pf in photo2tag:
                    photo2tag[pf].append(cat)
                else:
                    photo2tag[pf] = [cat]

        anno = [
            {'file_name': key, 'labels': [category[tag] for tag in val]}
            for key, val in tqdm(photo2tag.items())
        ]

        return anno

    @staticmethod
    def rep_mask(category, sim_thr=0.4, saved=True,
                 save_path='../datas/vis_rep/inputs/',
                 base_path='../datas/bases/'):
        print('calculating mask ...')
        all_sim = DH.loadJson(base_path + 'vis_sim_dict')
        repsnum = len(category)
        mask = np.zeros((repsnum, repsnum), int)
        for tag1 in tqdm(category):
            for tag2 in category:
                if tag1 == tag2:
                    continue

                if all_sim[tag1][tag2] >= sim_thr:
                    mask[category[tag1]][category[tag2]] = 1

        if saved:
            DH.savePickle(
                mask, '{0:0=2}.pickle'.format(int(sim_thr * 10)), save_path
            )

        return mask

    @staticmethod
    def down_anno(category, rep_category, phase='train',
                  base_path='../datas/bases/'):
        print('preparing dataset: {0} ...'.format(phase))
        t2f = DH.loadJson('down_tag2file', base_path)
        t2f = t2f[phase]
        down = set(category) - set(rep_category)
        photo2tag = {}
        for cat in down:
            for pf in t2f[cat]['filename']:
                if pf in photo2tag:
                    photo2tag[pf].append(cat)
                else:
                    photo2tag[pf] = [cat]

        anno = [
            {'file_name': key, 'labels': [category[tag] for tag in val]}
            for key, val in tqdm(photo2tag.items())
        ]

        return anno

    @staticmethod
    def _down_mask(rep_category, local_category, sim_thr=0.4,
                  saved=True, save_path='../datas/vis_down/inputs/',
                  base_path='../datas/bases/'):
        print('calculating mask ...')
        all_sim = DH.loadJson(base_path + 'vis_sim_dict')
        category = sorted(set(local_category) - set(rep_category))
        repsnum = len(local_category)
        mask = np.zeros((repsnum, repsnum), int)
        for tag1 in tqdm(category):
            for tag2 in category:
                if tag1 == tag2:
                    continue

                if all_sim[tag1][tag2] >= sim_thr:
                    mask[local_category[tag1]][local_category[tag2]] = 1

        if saved:
            DH.savePickle(
                mask, '{0:0=2}.pickle'.format(int(sim_thr * 10)), save_path
            )

        return mask

    @staticmethod
    def down_mask(rep_category, local_category, sim_thr=0.4,
                  saved=True, save_path='../datas/vis_down/inputs/',
                  base_path='../datas/bases/'):
        print('calculating mask ...')
        all_sim = DH.loadJson(base_path + 'vis_sim_dict')
        gsp_dict = DH.loadPickle('geospatial_df_area16_wocoth', base_path)
        gsp_dict = gsp_dict.to_dict('index')
        rep_dict = DH.loadPickle('rep_df_area16_wocoth', base_path)
        rep_dict = rep_dict.to_dict('index')

        comb_dict = {key: [] for key in local_category}
        for tag1 in local_category:
            for tag2 in local_category:
                if tag1 == tag2:
                    continue

                if all_sim[tag1][tag2] >= sim_thr:
                    comb_dict[tag1].append(tag2)

        # ---------------------------------------------------------------------
        lc = local_category
        down = sorted(list(set(lc) - set(rep_category)))
        tagsnum = len(lc)
        gspkeys = set(list(gsp_dict.keys()))
        lcset = set(lc)
        repset = set(rep_category)

        mask = np.zeros((tagsnum, tagsnum), int)
        for tag in tqdm(down):
            flgs = {tag}
            prev = set()
            checked = set()
            while flgs:
                for item in flgs:
                    checked.add(item)
                    prev = prev | set(gsp_dict[item]['representative'])

                prev = prev & lcset
                flgs = (prev - checked) & gspkeys

            exprev = set()
            for ptag in prev:
                if ptag in comb_dict:
                    exprev = exprev | (set(comb_dict[ptag]) & repset)

            prev = prev | exprev

            for ptag in prev:
                mask[lc[tag]][lc[ptag]] = 1
                for ctag in comb_dict[ptag]:
                    mask[lc[tag]][lc[ctag]] = 1

                temp = set(rep_dict[ptag]['down']) & lcset
                for ttag in temp:
                    if ttag != tag:
                        mask[lc[tag]][lc[ttag]] = 1

                    if ttag in rep_dict:
                        ttagdown = set(rep_dict[ttag]['down']) & lcset
                        for tdtag in ttagdown:
                            if tdtag != tag:
                                mask[lc[tag]][lc[tdtag]] = 1

            if tag in rep_dict:
                for rtag in rep_dict[tag]['down']:
                    if rtag in lcset and rtag != tag:
                        mask[lc[tag]][lc[rtag]] = 1

            for ctag in comb_dict[tag]:
                mask[lc[tag]][lc[ctag]] = 1

        np.fill_diagonal(mask, 0)

        if saved:
            DH.savePickle(
                mask, '{0:0=2}.pickle'.format(int(sim_thr * 10)), save_path
            )

        return mask

    # @staticmethod
    # def down_mask(rep_category, local_category, sim_thr=0.4,
    #               saved=True, save_path='../datas/vis_down/inputs/',
    #               base_path='../datas/bases/'):
    #     print('calculating mask ...')
    #     all_sim = DH.loadJson(base_path + 'vis_sim_dict')
    #     gsp_dict = DH.loadPickle('geospatial_df_area16_wocoth', base_path)
    #     gsp_dict = gsp_dict.to_dict('index')
    #     rep_dict = DH.loadPickle('rep_df_area16_wocoth', base_path)
    #     rep_dict = rep_dict.to_dict('index')

    #     lclist = sorted(list(local_category))
    #     comb_dict = {}
    #     for idx, tag1 in enumerate(tqdm(lclist[:-1])):
    #         for tag2 in lclist[idx + 1:]:
    #             if all_sim[tag1][tag2] < sim_thr:
    #                 continue

    #             if tag1 not in comb_dict:
    #                 comb_dict[tag1] = [tag2]
    #             else:
    #                 comb_dict[tag1].append(tag2)

    #             if tag2 not in comb_dict:
    #                 comb_dict[tag2] = [tag1]
    #             else:
    #                 comb_dict[tag2].append(tag1)

    #     # ---------------------------------------------------------------------
    #     lc = local_category
    #     down = sorted(list(set(lc) - set(rep_category)))
    #     tagsnum = len(lc)
    #     gspkeys = set(list(gsp_dict.keys()))
    #     lcset = set(lc)
    #     repset = set(rep_category)

    #     mask = np.zeros((tagsnum, tagsnum), int)
    #     for tag in tqdm(down):
    #         flgs = [tag]
    #         prev = []
    #         while flgs:
    #             temp = []
    #             for item in flgs:
    #                 temp.extend(list(
    #                     set(gsp_dict[item]['representative']) & lcset
    #                 ))

    #             prev.extend(temp)
    #             flgs = list(set(temp) & gspkeys)

    #         addprev = []
    #         for ptag in prev:
    #             if ptag in comb_dict:
    #                 addprev.extend(list(
    #                     set(comb_dict[ptag]) & repset
    #                 ))

    #         prev = list(set(prev) | set(addprev))

    #         for ptag in prev:
    #             mask[lc[tag]][lc[ptag]] = 1
    #             for ctag in comb_dict[ptag]:
    #                 mask[lc[tag]][lc[ctag]] = 1

    #             temp = list(
    #                 set(rep_dict[ptag]['down']) & lcset
    #             )
    #             for ttag in temp:
    #                 if ttag != tag:
    #                     mask[lc[tag]][lc[ttag]] = 1

    #                 if ttag in rep_dict:
    #                     ttagdown = list(
    #                         set(rep_dict[ttag]['down'])
    #                         & lcset
    #                     )
    #                     for tdtag in ttagdown:
    #                         if tdtag != tag:
    #                             mask[lc[tag]][lc[tdtag]] = 1

    #         if tag in rep_dict:
    #             for rtag in rep_dict[tag]['down']:
    #                 if rtag in lcset and rtag != tag:
    #                     mask[lc[tag]][lc[rtag]] = 1

    #         if tag in comb_dict:
    #             for ctag in comb_dict[tag]:
    #                 mask[lc[tag]][lc[ctag]] = 1

    #     for i in range(tagsnum):
    #         mask[i, i] = 0

    #     if saved:
    #         DH.savePickle(
    #             mask, '{0:0=2}.pickle'.format(int(sim_thr * 10)), save_path
    #         )

    #     return mask
