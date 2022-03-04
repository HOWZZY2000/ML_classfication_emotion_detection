import sys
sys.path.append("..")
from Preprocessing.Missing_value_handling import missing_value_handling
from Preprocessing.Scale import scale
from Utili.scatter_plot import *
from Settings import *

class Preprocessor:
    """
    A class used for loading and preprocessing data
    """
    def __init__(self, path=[GSR]) -> None:
        assert(all(map(Path.exists, path)))
        df = pd.concat(map(pd.read_csv, path), axis=1) # combining all data files
        self._df = df.loc[:, ~df.columns.duplicated()] # remove duplicated columns

    def get_x(self):  # training data column
        return self._df.iloc[:, 1:]

    def get_y(self):  # target column
        return self._df.iloc[:,0]

    def __str__(self):
        return self._df.to_string()

    def exam(self, columns) -> None:
        make_plot(self.get_x()[columns], self.get_y())

    def missing_value_handling(self, index=None, full=True, columns=[None]) -> None:
        if full:
            if index is None:
                missing_value_handling(self._df)
            else:
                missing_value_handling(self._df, index)
        else:
            missing_value_handling(self._df[columns], index)

    def scale(self, index=None, full=True, columns=[None]):
        if full:
            columns = self._df.columns
            if index is None:
                self._df = pd.DataFrame(scale(self._df), columns=columns)
            else:
                self._df = pd.DataFrame(scale(self._df, index), columns=columns)
        else:
            self._df[columns] = pd.DataFrame(scale(self._df[columns], index), columns=columns)








