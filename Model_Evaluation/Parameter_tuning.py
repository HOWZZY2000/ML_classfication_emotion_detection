import sys
sys.path.append("..")
from Utili.save_load import *
from Model_Evaluation.Parameters import *
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import seaborn as sns

class parameter_tuning():
    """
    A class used for tuning parameters and performing statistical testing
    """
    def __init__(self, pr):
        self.data = pr
        self.X_train = self.X_test = self.Y_train = \
        self.Y_test = self.search = self.estimator = \
        self.current_estimator = self.param = None

    def train_split(self, size=0.2):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.data.get_x(),
                                                                                self.data.get_y(),
                                                                                test_size=size)
    def report(self):
        self.search_check()
        y_true, y_pred = self.Y_test, self.search.predict(self.X_test)
        print(classification_report(y_true, y_pred))

    def build(self, index=0):
        if index == 0:  # svc estimator
            self.current_estimator = "SVC"
            self.estimator = SVC(probability=True)
            self.param = full_svm_param_grid
        elif index == 1:  # random forest estimator
            self.current_estimator = "RF"
            self.estimator = RandomForestClassifier()
            self.param = full_rf_param_grid
        elif index == 2: # KNN
            self.current_estimator = "KNN"
            self.estimator = KNeighborsClassifier()
            self.param = full_knn_param_grid
        else:
            raise ValueError('index does not exist')

    def halving_grid_search(self):
        self.estimator_check()
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
        self.search = HalvingGridSearchCV(estimator=self.estimator,
                                          param_grid=self.param, scoring=list_of_scoring,
                                          refit="f1", cv=cv)
        self.search.fit(self.data.get_x(), self.data.get_y())
        self.print_result()

    def grid_search(self): # evaluated using f1_micro score
        self.estimator_check()
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
        self.search = GridSearchCV(estimator=self.estimator,
                                   param_grid=self.param, scoring=list_of_scoring,
                                   refit="f1", cv=cv)
        self.search.fit(self.data.get_x(), self.data.get_y())
        self.print_result()

    def print_result(self):
        self.search_check()
        means = self.search.cv_results_["mean_test_f1"]
        stds = self.search.cv_results_["std_test_f1"]
        print()
        print("Best parameters set found on development set:")
        print(self.search.best_params_)
        print("Grid scores on development set:")
        for mean, std, params in zip(means, stds, self.search.cv_results_["params"]):
            print("%.3f (+/-%.3f) for %r" % (mean, std * 2, params))

    def dependency_plot(self, score):  # plot 30 examples of dependency between cv fold and f1 scores
        self.search_check()
        result_df = pd.DataFrame(self.search.cv_results_)
        result_df = result_df.set_index(result_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("kernel")
        model_scores = result_df.filter(regex=r"split\d*_test_" + score)

        fig, ax = plt.subplots()
        sns.lineplot(
            data=model_scores.transpose().iloc[:30],
            dashes=False,
            palette="Set1",
            marker="o",
            alpha=0.5,
            ax=ax,
        )
        ax.set_xlabel("CV test fold", size=12, labelpad=10)
        ax.set_ylabel(score + "score", size=12)
        ax.tick_params(bottom=True, labelbottom=False)
        plt.show()

    def save(self):
        save(self.search.best_params_, self.current_estimator)

    def estimator_check(self):
        if self.estimator is None or self.param is None:
            raise ValueError('Estimator or param is empty, must run build first')

    def search_check(self):
        if self.search is None:
            raise ValueError('Search is empty, must run search first')

