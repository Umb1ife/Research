import pandas as pd
from mmm import DataHandler as DH
from tqdm import tqdm


def geodata_format():
    local_dict = DH.loadPickle(
        '../datas/backup/def_georep/local_df_area16_wocoth_refined.pickle'
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


if __name__ == "__main__":
    geodata_format()
