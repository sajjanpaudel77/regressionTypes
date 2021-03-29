# decision Tree algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('resources/pos_salary.csv')

real_x = data.iloc[:, 1:2].values
real_y = data.iloc[:, 2:3].values

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(real_x, real_y)

y_pred = dtr.predict([[6]])
print(y_pred)

x_grid = np.arange(min(real_x), max(real_x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(real_x, real_y, color = 'Blue')
plt.plot(x_grid, dtr.predict(x_grid), color = 'Red')
plt.title('DTR')
plt.xlabel('Position Level')
plt.xlabel('Salary')
plt.show()