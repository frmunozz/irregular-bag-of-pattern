import os
import sys
import numpy as np
import pandas as pd


main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_path)

if __name__ == "__main__":
    data_path = os.path.join(main_path, "data", "bop_sparse_repr")
    df = pd.read_csv(os.path.join(data_path, "metadata.csv"))
