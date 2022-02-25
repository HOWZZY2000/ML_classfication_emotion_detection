from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from Settings import *
from Preprocessing import Preprocessor


# print(df.iloc[:, 1:].head(5))

# print(df.iloc[:, 0].head(5))
data = Preprocessor()
data.missing_value_handling()

X_train, X_test, Y_train, Y_test = train_test_split(data.df.iloc[:, 1:], data.df.iloc[:, 0], test_size=0.3)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))