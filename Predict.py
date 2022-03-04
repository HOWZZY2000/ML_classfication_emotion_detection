from Classifiers.SVM import SVM
from Preprocessing.Preprocessor import Preprocessor

data = Preprocessor()
data.missing_value_handling(2)
data.scale(6)
data.exam(["GSR_Mean", "SD"])
# Predictor = SVM(data)
# Predictor.train()
# print(Predictor.fit())