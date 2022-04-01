from Classifiers.RF import RF
from Classifiers.SVM import SVM
from Preprocessing.Preprocessor import Preprocessor
from Model_Evaluation.Parameter_tuning import parameter_tuning
from Model_Evaluation.Feature_selection import pca
from Utili.save_load import *
from Classifiers.my_KNN import *
from Classifiers.KNN import KNN
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = Preprocessor()
    data.missing_value_handling(2)
    data.scale(8)
    # pca(3, data.get_x())

    # m = m_KNN()
    # train_X, train_Y, test_X, test_Y = m_KNN.train_split(data._df)
    # print(m_KNN.score(m_KNN.fit(train_X, train_Y, test_X, 3), test_Y))

    # examining outliers
    # data.exam(["GSR_Mean", "SD"])

    # parameter tuning
    # parameter_selection = parameter_tuning(data)
    # parameter_selection.build(1)
    # parameter_selection.grid_search()
    # parameter_selection.dependency_plot("f1")

    # RF
    # Predictor = RF(data)
    # Predictor.train()
    # print(Predictor.fit())

    # SVM
    # Predictor = SVM(data)
    # Predictor.train()
    # print(Predictor.fit())

    # Save & Load
    # save(Predictor.model)
    # model = load()
    # print(type(model))

