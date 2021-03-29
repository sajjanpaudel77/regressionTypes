import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('resources/pos_salary.csv')

real_x = data.iloc[:, 1:2].values
real_y = data.iloc[:, 2].values

reg = RandomForestRegressor(n_estimators=300, random_state=0)
reg.fit(real_x, real_y)

y_pred = reg.predict([[6.5]])
print(y_pred)

x_grid = np.arange(min(real_x), max(real_x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(real_x, real_y, color = 'red')
plt.plot(x_grid, reg.predict(x_grid), color = 'Blue')
plt.title('Random Forest')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

