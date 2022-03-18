import sys
sys.path.append("..")
from Settings import *
from sklearn.metrics import make_scorer, cohen_kappa_score

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 50, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# The function to measure the quality of a split
criterion = ["gini", "entropy"]
# Create the random grid
full_rf_param_grid = dict(n_estimators=n_estimators,
               max_depth=max_depth,
               min_samples_split=min_samples_split,
               min_samples_leaf=min_samples_leaf,
               criterion=criterion)

full_svm_param_grid = [{"kernel": ["linear"], "degree": [1, 2, 3, 4, 5, 6], "C": np.logspace(-3, 2, 6)},
                       {"kernel": ["poly"], "C": np.logspace(-3, 2, 6), "gamma": np.logspace(-3, 2, 6)},
                       {"kernel": ["rbf"], "C": np.logspace(-3, 2, 6), "gamma": np.logspace(-3, 2, 6)}]

# hyper-parameters for KNN
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
full_knn_param_grid= dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

list_of_scoring = {"f1": "f1_micro", "accuracy": "balanced_accuracy", "kappa": make_scorer(cohen_kappa_score)}

# parameter already found with only GSR
svc_best_param = [{"kernel": ["linear"], "degree": [1], "C": [3, 5]}]