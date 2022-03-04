import sys
sys.path.append("..")
from Settings import *


def missing_value_handling(df, index=1) -> None:
    if df.isna().values.any():
        if index == 0:  # drop the row
            df.dropna(inplace=True)
        elif index == 1:  # using mean
            for i in df.columns[df.isna().any()].tolist():
                df[i] = df[i].replace(np.NaN, df[i].mean())
        elif index == 2:  # using median
            for i in df.columns[df.isna().any()].tolist():
                df[i] = df[i].replace(np.NaN, df[i].median())
        elif index == 3:  # using mode
            for i in df.columns[df.isna().any()].tolist():
                df[i] = df[i].replace(np.NaN, df[i].mode())
        elif index == 4:  # using 0
            df.fillna(0)
        elif index == 5:  # using last observation carried forward
            for i in df.columns[df.isna().any()].tolist():
                df[i] = df[i].fillna(method="ffill")
        elif index == 6:  # use interpolation
            for i in df.columns[df.isna().any()].tolist():
                df[i] = df[i].interpolate(method='linear', limit_direction='forward', axis=0)
        # elif index == 7: # using
        else:
            raise ValueError('index does not exist')
    assert (1 - df.isna().sum().sum())
