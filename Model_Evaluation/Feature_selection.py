import sys
sys.path.append("..")
from Settings import *
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

def importance_feature_selection(data):
    # selecting based on importance
    ridge = RidgeCV(alphas=np.logspace(-10, 10, num=21)).fit(data.X, data.Y)
    importance = np.abs(ridge.coef_)
    feature_names = data.df.columns[1:]
    # importance above the sixth parameter
    threshold = np.sort(importance)[-6] + 0.01
    sfm = SelectFromModel(ridge, threshold=threshold).fit(data.X, data.Y)
    # print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
    return feature_names[sfm.get_support()]

def pca(num, data):
    selection = PCA(n_components=num)
    selection.fit(data)
    print("Percentage of variance explained by each of the selected components: ")
    print(selection.explained_variance_ratio_)
    return pd.DataFrame(selection.transform(data))

