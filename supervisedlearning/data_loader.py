# this file loads the cleaned file data so the model can train on it

import pandas as pd
import os

def load_data():
    current_dir= os.path.dirname(__file__)
    data_path= os.path.join(current_dir, "..", "data","cleaned_data123.csv")
    df = pd.read_csv(data_path)
    return df