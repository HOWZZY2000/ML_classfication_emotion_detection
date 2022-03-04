import sys
sys.path.append("..")
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from Settings import *


def scale(df, index=0) -> np.array:
    if index == -1: # don't use any Normalization
        pass
    elif index == 0:  # using Z-Score Normalization with Mean-Standard Deviation
        df = StandardScaler().fit_transform(df)
    elif index == 1: # using Min-Max Normalization
        df = MinMaxScaler().fit_transform(df)
    elif index == 2: # using Abs Max Normalization
        df = MaxAbsScaler().fit_transform(df)
    elif index == 3: # using yeo-johnson power transformation
        df = PowerTransformer(method="yeo-johnson").fit_transform(df)
    elif index == 4: # using box-cox power transformation
        assert((df.values < 0).any())
        df = PowerTransformer(method="box-cox").fit_transform(df)
    elif index == 5: # using quantile transformation with uniform distribution
        df = QuantileTransformer(output_distribution="uniform").fit_transform(df)
    elif index == 6: # using quantile transformation with gaussian distribution
        df = QuantileTransformer(output_distribution="normal").fit_transform(df)
    elif index == 7: # using sample-wise L2 normalization
        df = Normalizer().fit_transform(df)
    elif index == 8: # using robust scaling normalization
        df = RobustScaler(quantile_range=(25, 75)).fit_transform(df)
    else:
        raise ValueError('index does not exist')
    return df