from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _map_diag(self, val):
        try:
            val = float(val)
        except:
            return 0
        if 1 <= val < 140: return 1
        elif 140 <= val < 240: return 2
        elif 240 <= val < 280: return 3
        elif 280 <= val < 290: return 4
        elif 290 <= val < 320: return 5
        elif 320 <= val < 390: return 6
        elif 390 <= val < 460: return 7
        elif 460 <= val < 520: return 8
        elif 520 <= val < 580: return 9
        elif 580 <= val < 630: return 10
        elif 630 <= val < 680: return 11
        elif 680 <= val < 710: return 12
        elif 710 <= val < 740: return 13
        elif 740 <= val < 760: return 14
        elif 760 <= val < 780: return 15
        elif 780 <= val < 800: return 16
        elif 800 <= val < 1000: return 17
        return 0

    def _map_age(self, val):
        # Accept both [60-70) or raw numeric like 63
        if isinstance(val, str) and "[" in val:
            val = val.strip("[]").split("-")
            return (int(val[0]) + int(val[1])) // 2
        else:
            val = float(val)
            val = np.clip(val, 0, 100)
            return int(((val // 10) * 10) + 5)

    def _map_insulin(self, val):
        return {'No': -2, 'Down': -1, 'Steady': 0, 'Up': 1}.get(val, -2)

    def _map_race(self, val):
        return {
            'Caucasian': 0, 'AfricanAmerican': 1,
            'Asian': 2, 'Hispanic': 3, 'Other': 4
        }.get(val, 4)

    def transform(self, X):
        X = X.copy()
        X['diag_1'] = X['diag_1'].apply(self._map_diag)
        X['diag_2'] = X['diag_2'].apply(self._map_diag)
        X['diag_3'] = X['diag_3'].apply(self._map_diag)
        X['age'] = X['age'].apply(self._map_age)
        X['insulin'] = X['insulin'].apply(self._map_insulin)
        X['race'] = X['race'].apply(self._map_race)
        return X
