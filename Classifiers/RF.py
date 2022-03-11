import sys
sys.path.append("..")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class RF:
    """
    Random Forest
    """

    def __init__(self, pr) -> None:
        self.data = pr

    # def feature_selection(self):
    #     self.data.df.loc[:, importance_feature_selection(self.data)]

    def train(self) -> None:
        self.X_train,self. X_test, self.Y_train, self.Y_test = train_test_split(self.data.get_x(),
                                                            self.data.get_y(), test_size=0.3)
        # building classifier
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.Y_train)

    def fit(self) -> float:
        return self.model.score(self.X_test, self.Y_test)