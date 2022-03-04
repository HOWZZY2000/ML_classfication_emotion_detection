import sys
sys.path.append("..")
from Model_Evaluation.Feature_selection import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class SVM:
    """
    Support Vector Machine
    """
    # data, model, X_train, X_test, Y_train, Y_test = (None,) * 6
    def __init__(self, pr) -> None:
        self.data = pr

    # def feature_selection(self):
    #     self.data.df.loc[:, importance_feature_selection(self.data)]

    def train(self) -> None:
        self.X_train,self. X_test, self.Y_train, self.Y_test = train_test_split(self.data.get_x(),
                                                            self.data.get_y(), test_size=0.3)
        # scaling the data
        self.model = make_pipeline(StandardScaler(), SVC(gamma='scale'))
        self.model.fit(self.X_train, self.Y_train)

    def fit(self) -> float:
        return self.model.score(self.X_test, self.Y_test)