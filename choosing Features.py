import pandas as pd
# this is important as we can choose multiple columns as we want to chose
data = pd.read_csv('resources/melbourne_2.csv')
data_features = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude']

x = data[data_features]
print(x)
print(x.describe())