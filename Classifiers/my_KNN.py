import operator
import pandas as pd
import numpy as np

class m_KNN:
    """
    develop my own version of the K Nearest Neighbors
    """

    def __init__(self) -> None:
        self.data = None

    # runtime of nk where n is traning sample size, k is testing sample size
    @staticmethod
    def fit(train_X, train_Y, test_X, k):
        test_Y = pd.DataFrame(columns=['result'])
        len = 0
        for index, row in test_X.iterrows():
            diffMat = train_X.sub(row, axis=1)
            sqDiffMat = diffMat ** 2
            sqDistances = sqDiffMat.sum(axis=1)
            sortedDistIndices = sqDistances.argsort()
            classCount = {}
            for i in range(k):
                voteIlabel = train_Y.iat[sortedDistIndices.iat[0]]
                classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            test_Y.loc[len] = sortedClassCount[0][0]
            len += 1
        return test_Y

    @staticmethod
    def train_split(data, test_size=0.3):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test size has to be in range of (0, 1)")
        test = data.sample(frac=test_size)
        train = data.iloc[~data.index.isin(test.index)]
        return train.iloc[:, 1:], train.iloc[:, 0], test.iloc[:, 1:], test.iloc[:, 0]

    @staticmethod
    def score(m_test_Y, test_Y):
        if m_test_Y.shape[0] != test_Y.shape[0]:
            raise ValueError("Number of rows should be same")
        return (m_test_Y.to_numpy().reshape(-1) == test_Y.to_numpy()).sum() / test_Y.shape[0]









