import numpy as np
# from .datahandler import DataHandler as DH
from collections import Counter
from scipy.stats import chi2
from sklearn.cluster import MeanShift
# from tqdm import tqdm


class MeanShiftDE:
    def __init__(self):
        self._ellipses = None

    def get_ellipse(self, datas, p=0.95, bandwidth=5, bin_seeding=True):
        c = np.sqrt(chi2.ppf(p, 2))
        self._ellipses = self._culculate_ellipse(
            c, datas, bandwidth, bin_seeding
        )

        return self._ellipses

    def _culculate_ellipse(self, c, locates, bandwidth, bin_seeding):
        '''
        データを元にあるタグのクラスタとなる楕円を計算
        '''
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
        ms.fit(locates)

        labels = ms.labels_
        count_dict = Counter(labels)

        ellipses = []
        for label, count in count_dict.items():
            if count < 0.05 * len(locates) or count < 2:
                continue

            center = ms.cluster_centers_[label]
            _datas = np.array(locates)[labels == label]
            _cov = np.cov(_datas[:, 0], _datas[:, 1])
            _lambdas, _ = np.linalg.eigh(_cov)
            _order = _lambdas.argsort()[::-1]
            _lambdas = _lambdas[_order]

            width, height = c * np.sqrt(abs(_lambdas))
            if _lambdas[0] - _lambdas[1] == 0:
                theta = 0
            elif _cov[0, 1] == 0:
                theta = np.arctan(np.sign(_lambdas[0] - _lambdas[1]) * np.inf)
            else:
                theta = np.arctan((_lambdas[0] - _lambdas[1]) / _cov[0, 1])

            ellipses.append([center, width, height, theta])

        return ellipses

    def _rotate(self, origin, point, angle):
        '''
        originを中心とする座標系でangleだけ傾きを変えたときのpointの座標を返すメソッド
        '''
        ox, oy = origin
        px, py = point
        qx = np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

        return qx, qy

    def _in_ellipse(self, point, ellipse):
        '''
        あるpointがellipse内に存在するかどうかを判定するメソッド
        '''
        origin, width, height, theta = ellipse
        px, py = self._rotate(origin, point, -theta)

        if (width == 0 and px != 0) or (height == 0 and py != 0):
            return False
        else:
            width = np.inf if width == 0 else width
            height = np.inf if height == 0 else height

            return False if (px / width) ** 2 + (py / height) ** 2 > 1 \
                else True

    def is_positive(self, point, datas=None,
                    p=0.95, bandwidth=5, bin_seeding=True):
        if datas is not None:
            self._ellipses = self.get_ellipse(datas, p, bandwidth, bin_seeding)
        else:
            if self._ellipses is None:
                raise ValueError('distribution is not calculated. \
                    assign arguments "datas" or first call get_ellipse().')

            # print('use last ellipses')

        for elp in self._ellipses:
            flg = self._in_ellipse(point, elp)
            if flg:
                return flg

        return False
