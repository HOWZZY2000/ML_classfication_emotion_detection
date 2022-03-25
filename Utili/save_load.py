import sys
sys.path.append("..")
from Settings import *
import pickle
import datetime


def save(model, file_name="Model.pkl") -> None:
    today = datetime.datetime.strptime(datetime.datetime.date(datetime.datetime.now()), "%Y-%m-%d")
    with open(Model_folder / today + file_name, 'wb') as file:
        pickle.dump(model, file)

def load(file_name="Model.pkl"): # return sklearn classifier
    with open(Model_folder / file_name, 'rb') as file:
        model = pickle.load(file)
    return model