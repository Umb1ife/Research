from geodown_training import limited_category
from mmm import DataHandler as DH
from mmm import MeanShiftRefiner
from tqdm import tqdm


if __name__ == "__main__":
    lda = DH.loadPickle('local_df_area16_wocoth_base.pickle',
                        '../datas/geo_down/inputs')

    for idx, data in tqdm(lda.iterrows(), total=lda.shape[0]):
        msr = MeanShiftRefiner(data['geo'])
        rl = [item for item in data['geo'] if msr.check(item)]
        lda.at[idx, 'geo'] = rl

    lda_new = '../datas/geo_down/inputs/local_df_area16_wocoth_new.pickle'
    DH.savePickle(lda, lda_new)
    rep_category = {'lasvegas': 0, 'newyorkcity': 1, 'seattle': 2}
    category = limited_category(rep_category, lda_new)

    print('finish.')
