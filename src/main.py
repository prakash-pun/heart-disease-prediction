import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from utils import get_csv_file
from scale import scale_minmax


scaled_data = scale_minmax()

print(scaled_data.head()) 