from Classifiers.RF import RF
from Classifiers.SVM import SVM
from Preprocessing.Preprocessor import Preprocessor
from Model_Evaluation.Parameter_tuning import parameter_tuning
from Utili.save_load import *

# examining outliers
# data.exam(["GSR_Mean", "SD"])

data = Preprocessor()
data.missing_value_handling(2)
data.scale(8)

# parameter tuning
# parameter_selection = parameter_tuning(data)
# parameter_selection.grid_search()
# parameter_selection.dependency_plot("f1")

# RF
# Predictor = RF(data)
# Predictor.train()
# print(Predictor.fit())

# SVM
Predictor = SVM(data)
Predictor.train()
print(Predictor.fit())

# Save & Load
# save(Predictor.model)
# model = load()
# print(type(model))

