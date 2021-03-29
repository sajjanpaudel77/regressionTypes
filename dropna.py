import pandas as pd
import matplotlib.pyplot as plt

meldata = pd.read_csv("resources/melbourne_2.csv")
print(meldata.columns)

print(meldata.dropna(axis = 0).head(10)) # dropna drops the missing value and just displays whose all values are given
