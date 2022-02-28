from Classifiers.SVM import SVM
from Preprocessing.preprocessor import Preprocessor


data = Preprocessor()
data.missing_value_handling(2)
Predictor = SVM(data)
Predictor.train()
print(Predictor.fit())