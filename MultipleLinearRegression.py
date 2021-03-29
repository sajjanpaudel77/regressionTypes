import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

data = pd.read_csv('resources/vishwajeetmultiple.csv')
# here newyork, california and florida is categorial data

real_x = data.iloc[:,0:4].values  # multiple input
real_y = data.iloc[:, 4].values # one output


ct = ColumnTransformer([('oneHT',OneHotEncoder(categories='auto'),[3])],remainder='passthrough')

real_x = ct.fit_transform(real_x)

# part 2 starts
real_x = real_x[:, 1:]

training_x, testing_x, training_y, testing_y = train_test_split(real_x, real_y, test_size=0.20, random_state= 0) # 80% training automatically

MLR = LinearRegression()

MLR.fit(training_x, training_y)

pred_y = MLR.predict(testing_x)
print(pred_y) # this is predicted output

print(testing_y) # this is actual output

##
##print(MLR.coef_) # prints the coefficient i.e b1
##print(MLR.intercept_) # prints the intercept
## this is not well predicted look at the difference of actual and predicted values
# so we gonna use backward elimination method by chosing highest value
# and comparing with predector
# if p > 5% , chose another p and repeat until condition is false



real_x = np.append(arr= np.ones((50,1)).astype(int),values=real_x, axis =1)
print(real_x) # now x0 = 1


x_opt = np.array(real_x[:, [0,1,2,3,4,5]], dtype = float) # 5th is profit value

regressor_OLS = sm.OLS(endog= real_y, exog= x_opt).fit() # now model is fit
print(regressor_OLS.summary()) #prints the summary

# watch out p>|t|
#x2 has the big so we remove index value 2, then index value 1, then index value of 2, index value 2
x_opt = np.array(real_x[:, [0,3]], dtype = float)
regressor_OLS = sm.OLS(endog= real_y, exog= x_opt).fit() # now model is fit
print(regressor_OLS.summary()) #prints the summary
# then we finally get the value p < 0.05, then we can find most significant independent variable i.e. const and x1
# column 0 and column 3 will give the optimized result for this algorithm

