import pandas as pd
import numpy as np
from pathlib import Path

# Save global constant paths
data_folder = Path("dataset/")
ECG = data_folder / "ECG_FeaturesExtracted.csv"
ET = data_folder / "EyeTracking_FeaturesExtracted.csv"
GSR = data_folder / "GSR_FeaturesExtracted.csv"

# Only for printing purpose
pd.set_option("display.max_rows", None, "display.max_columns", None)