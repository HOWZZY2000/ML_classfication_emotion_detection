import sys
sys.path.append("..")
from Settings import *

class Preprocessor:
    """
    A class used for loading and pre-processing data
    """
    df = None
    def __init__(self, path=[ECG, ET, GSR]) -> None:
        df = pd.concat(map(pd.read_csv, path), axis=1) # combining all data files
        self.df = df.loc[:, ~df.columns.duplicated()] # remove duplicated columns

    def exam(self) -> None:
        return

    def missing_value_handling(self, index=1) -> None:
        if self.df.isna().values.any():
            if index == 0:  # drop the row
                self.df.dropna(inplace=True)
            elif index == 1:  # using mean
                for i in self.df.columns[self.df.isna().any()].tolist():
                    self.df[i] = self.df[i].replace(np.NaN, self.df[i].mean())
            elif index == 2:  # using median
                for i in self.df.columns[self.df.isna().any()].tolist():
                    self.df[i] = self.df[i].replace(np.NaN, self.df[i].median())
            elif index == 3:  # using mode
                for i in self.df.columns[self.df.isna().any()].tolist():
                    self.df[i] = self.df[i].replace(np.NaN, self.df[i].mode())
            elif index == 4: # using 0
                self.df.fillna(0)
            else:
                raise ValueError('index does not exist')



