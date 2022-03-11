import sys
sys.path.append("..")
from Settings import *
import pickle

def save(model, file_name="Model.pkl") -> None:
    with open(Model_folder / file_name, 'wb') as file:
        pickle.dump(model, file)

def load(file_name="Model.pkl"): # return sklearn classifier
    with open(Model_folder / file_name, 'rb') as file:
        model = pickle.load(file)
    return model