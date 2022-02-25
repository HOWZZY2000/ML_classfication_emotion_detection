import sys
sys.path.append("..")
from Settings import *
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel

def importance_feature_selection(data):
    # selecting based on importance
    ridge = RidgeCV(alphas=np.logspace(-10, 10, num=21)).fit(data.df.iloc[:, 1:], data.df.iloc[:, 0])
    importance = np.abs(ridge.coef_)
    feature_names = data.df.columns[1:]
    # importance above the sixth parameter
    threshold = np.sort(importance)[-6] + 0.01
    sfm = SelectFromModel(ridge, threshold=threshold).fit(data.df.iloc[:, 1:], data.df.iloc[:, 0])
    # print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
    return feature_names[sfm.get_support()]


