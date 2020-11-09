import glob
import numpy as np
import pandas as pd
from .datahandler import DataHandler as DH
from .mmfunction import loading_animation
from tqdm import tqdm


class PrepareVis:
    @loading_animation('category')
    def _category(self, start, limited):
        for tag in self._represent_dict:
            if len(self._represent_dict[tag]['down']) == 1:
                self._remove_tags.append(tag)

        self._remove_tags = list(set(self._remove_tags))

        prev = []
        self._representations = [start]
        ldkeys = list(self._local_dict.keys())
        while (set(prev) != set(self._representations)):
            temp = []
            for tag in self._representations:
                temp.extend(self._represent_dict[tag]['down'])

            temp = list(set(temp))
            prev = self._representations.copy()

            for tag in temp:
                self._representations.extend(
                    self._local_dict[tag]['representative']
                )

            self._representations = list(
                set(self._representations)
                - set(self._remove_tags)
                - set(ldkeys)
            )

        self._representations = set(self._representations) if limited is None \
            else set(self._representations) & set(limited)

        down_tags = []
        for tag in self._representations:
            down_tags.extend(self._represent_dict[tag]['down'])

        onehop = list(
            set(down_tags)
            & set(list(self._represent_dict.keys())) - set(self._remove_tags)
        )
        for tag in onehop:
            down_tags.extend(self._represent_dict[tag]['down'])

        down_tags = set(down_tags)
        twohop = []
        for tag in down_tags:
            twohop.extend(self._local_dict[tag]['representative'])

        twohop = set(twohop) - set(self._remove_tags) - down_tags - set(ldkeys)
        twohop = twohop if limited is None else twohop & set(limited)

        self._representations = self._representations | twohop
        self._local_tags = sorted(list(self._representations | down_tags))
        self._representations = sorted(list(self._representations))
        self._down_tags = sorted(list(down_tags))

        # ---------------------------------------------------------------------
        # save
        representations = {
            key: idx for idx, key in enumerate(self._representations)
        }
        local_tags = {key: idx for idx, key in enumerate(self._local_tags)}

        # save fine_tuning category
        DH.saveJson(
            representations, 'category.json',
            self._outpath + 'fine_tuning/inputs'
        )

        # save gcn category
        DH.saveJson(
            representations, 'upper_category.json',
            self._outpath + 'gcn/inputs'
        )
        DH.saveJson(local_tags, 'category.json', self._outpath + 'gcn/inputs')

    def _mask(self, sim_thr):
        left = [item[0][0] for item in self._all_sim]
        right = [item[0][1] for item in self._all_sim]
        sim = [item[1] for item in self._all_sim]
        representations = self._representations
        local_tags = self._local_tags

        comb_list = pd.DataFrame(
            zip(left, right, sim), columns=['left', 'right', 'sim']
        )
        comb_list = comb_list[
            (comb_list['left'].isin(local_tags))
            & (comb_list['right'].isin(local_tags))
        ]
        comb_list = comb_list[comb_list.sim >= sim_thr]
        comb_list = comb_list[['left', 'right']].values.tolist()

        comb_dict = list(set([item for tpl in comb_list for item in tpl]))
        comb_dict = {key: [] for key in comb_dict}
        for l, r in comb_list:
            comb_dict[l].append(r)
            comb_dict[r].append(l)
        # -------------------------------------------------------------------------
        # mask_の作成

        # for GCN
        tag2index = {cat: idx for idx, cat in enumerate(local_tags)}
        tagsnum = len(local_tags)
        gcn_mask = np.zeros((tagsnum, tagsnum), int)
        ldkeys = list(self._local_dict.keys())

        print('Step. {0}/9'.format(int(sim_thr * 10)))
        for tag in tqdm(self._down_tags):
            flgs = [tag]
            prev = []
            while flgs:
                temp = []
                for item in flgs:
                    temp.extend(list(
                        set(self._local_dict[item]['representative'])
                        & set(local_tags)
                    ))

                prev.extend(temp)
                flgs = list(set(temp) & set(ldkeys))

            addprev = []
            for ptag in prev:
                if ptag in comb_dict:
                    addprev.extend(list(
                        set(comb_dict[ptag]) & set(self._representations)
                    ))

            prev = list(set(prev) | set(addprev))

            for ptag in prev:
                gcn_mask[tag2index[tag]][tag2index[ptag]] = 1
                for ctag in comb_dict[ptag]:
                    gcn_mask[tag2index[tag]][tag2index[ctag]] = 1

                temp = list(
                    set(self._represent_dict[ptag]['down']) & set(local_tags)
                )
                for ttag in temp:
                    if ttag != tag:
                        gcn_mask[tag2index[tag]][tag2index[ttag]] = 1

                    if ttag in self._represent_dict:
                        ttagdown = list(
                            set(self._represent_dict[ttag]['down'])
                            & set(local_tags)
                        )
                        for tdtag in ttagdown:
                            if tdtag != tag:
                                gcn_mask[tag2index[tag]][tag2index[tdtag]] = 1

            if tag in self._represent_dict:
                for rtag in self._represent_dict[tag]['down']:
                    if rtag in local_tags and rtag != tag:
                        gcn_mask[tag2index[tag]][tag2index[rtag]] = 1

            if tag in comb_dict:
                for ctag in comb_dict[tag]:
                    gcn_mask[tag2index[tag]][tag2index[ctag]] = 1

        for i in range(tagsnum):
            gcn_mask[i, i] = 0

        DH.savePickle(
            gcn_mask, '{0:0=2}.pickle'.format(int(sim_thr * 10)),
            self._outpath + 'gcn/inputs/comb_mask'
        )

        # for fine-tuning
        rep2index = {cat: idx for idx, cat in enumerate(representations)}
        repsnum = len(representations)
        ft_mask = np.zeros((repsnum, repsnum), int)

        for tag in representations:
            if tag in comb_dict:
                for ctag in comb_dict[tag]:
                    if ctag in representations:
                        ft_mask[rep2index[tag]][rep2index[ctag]] = 1

        for i in range(repsnum):
            ft_mask[i, i] = 0

        DH.savePickle(
            ft_mask, '{0:0=2}.pickle'.format(int(sim_thr * 10)),
            self._outpath + 'fine_tuning/inputs/comb_mask'
        )

    @loading_animation('Load Data')
    def _load_data(self, data_paths, outpath, lowerlimit_photo_n):
        self._local_dict = DH.loadPickle(data_paths['local_df'])
        self._local_dict = self._local_dict.to_dict('index')
        self._represent_dict = DH.loadPickle(data_paths['rep_df'])
        self._represent_dict = self._represent_dict.to_dict('index')
        self._all_sim = DH.loadPickle(data_paths['all_sim'])
        self._all_sim = self._all_sim.values.tolist()
        self._remove_tags = DH.loadPickle(data_paths['rep_photo'])
        self._remove_tags = self._remove_tags[
            self._remove_tags.photo_num < lowerlimit_photo_n
        ]['tag'].tolist()
        self._photo_directory = data_paths['photo_directory']

        self._outpath = outpath if outpath[-1:] == '/' else outpath + '/'

    def _anno(self, stage, phase):
        filepath = self._photo_directory['{0}_{1}'.format(stage, phase)]
        filepath = filepath if filepath[-1:] == '/' else filepath + '/'

        labels = self._representations if stage == 'rep' else \
            sorted(list(set(self._local_tags) - set(self._representations)))
        l2idx = self._representations if stage == 'rep' else self._local_tags
        l2idx = {lbl: idx for idx, lbl in enumerate(l2idx)}

        files = {}
        for label in tqdm(labels):
            temp = glob.glob(filepath + label + '/*jpg')
            for photo in temp:
                photo = photo.replace(filepath + label + '/', '')
                if photo in files:
                    files[photo]['labels'].append(l2idx[label])
                else:
                    files[photo] = {
                        'file_name': photo, 'labels': [l2idx[label]]
                    }

        return list(files.values())

    def _annotation(self):
        stages, phases = ['rep', 'local'], ['train', 'validate']
        cnt = 0
        for st in stages:
            opath = self._outpath + 'fine_tuning/inputs/' if st == 'rep' \
                else self._outpath + 'gcn/inputs/'
            for ph in phases:
                cnt += 1
                print('Step. {0}/4'.format(cnt))
                anno = self._anno(st, ph)
                DH.saveJson(anno, '{0}_anno.json'.format(ph), opath)

    def _images(self):
        '''
        各ラベル名のついたディレクトリ下にある画像についてtrain/validate以下に展開
        '''

    def _relationship(self):
        '''
        基本はrep_df_area16_wocoth.pickleの中身をdict化すればOK？
        '''

    def before_finetuning(self, data_paths={}, outpath='../datas/',
                          lowerlimit_photo_n=100, start='church',
                          limited_category=None):
        # load data
        self._load_data(data_paths, outpath, lowerlimit_photo_n)

        # make data
        self._category(start, limited_category)
        print('mask:')
        for i in range(1, 5):
            self._mask(i / 10)
        print('annotation:')
        self._annotation()

    def after_finetuning(self):
        '''
        finetuneで学習したモデルの重みをgcnのinputディレクトリに移動させる
        '''


if __name__ == "__main__":
    path_top = '../datas/prepare/'
    dp_dict = {
        'local_df': path_top + 'local_df_area16_wocoth.pickle',
        'rep_df': path_top + 'rep_df_area16_wocoth.pickle',
        'all_sim': path_top + 'all_sim.pickle',
        'rep_photo': path_top + 'rep_photo_num.pickle',
        'photo_directory': {
            'rep_train': path_top + 'rep_train',
            'rep_validate': path_top + 'rep_validate',
            'local_train': path_top + 'local_train',
            'local_validate': path_top + 'local_validate'
        }
    }
