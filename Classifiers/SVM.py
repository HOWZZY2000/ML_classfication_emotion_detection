import sys
sys.path.append("..")
from Model_Evaluation.Feature_selection import *
from Preprocessing.Preprocessing import Preprocessor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class SVM:
    """
    Support Vector Machine
    """
    data, model, X_train, X_test, Y_train, Y_test = (None,) * 6
    def __init__(self) -> None:
        self.data = Preprocessor()
        self.data.missing_value_handling(1)
        # data.df.loc[:, importance_feature_selection(data)]

    def train(self) -> None:
        self.X_train,self. X_test, self.Y_train, self.Y_test = train_test_split(self.data.df.iloc[:, 1:],
                                                            self.data.df.iloc[:, 0], test_size=0.4)
        self.model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.model.fit(self.X_train, self.Y_train)

    def fit(self) -> float:
        return self.model.score(self.X_test, self.Y_test)