import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('resources/pos_salary.csv')
## print(data)

real_x = data.iloc[:, 1:2].values # shows at array, but if we do iloc[:, 1], it will shows as vector
real_y = data.iloc[:, 2].values # dependent variable doesnotnot need to be in 2d so we say only iloc[:, 2]


linear_reg = LinearRegression()
linear_reg.fit(real_x, real_y) # training model

polynomial_reg = PolynomialFeatures(degree = 10)
real_x_poly = polynomial_reg.fit_transform(real_x)
## print(real_x_poly) , b0 =1 by default. now it is converted in degree of 2

# now we try to fit the polynomial independent value
polynomial_reg.fit(real_x_poly, real_y)


# now we have to apply regression over real_x_poly
linear_reg2 = LinearRegression()
linear_reg2.fit(real_x_poly, real_y)

plt.scatter(real_x, real_y, color = 'red')
plt.plot(real_x, linear_reg.predict(real_x), color = 'blue')
plt.title('Linear Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()  # this is linear model it is bad so we gonna different

plt.scatter(real_x, real_y, color = 'red')
plt.plot(real_x, linear_reg2.predict(real_x_poly), color = 'blue')
plt.title('Poly Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# now lets predict
# substitute for b0 b1 and x then predict y
print(linear_reg.predict([[10.5]])) # prints simple linear
print(linear_reg2.predict(polynomial_reg.fit_transform([[10.5]]))) # prints polynomial linear






