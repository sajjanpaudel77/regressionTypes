import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("resources/vishwajeet.csv")
print(data)

real_x = data.iloc[:,0:1].values # all rows but only the 1st column and values helps to show data in array
real_y = data.iloc[:,1:2].values # all rows but only the 2nd column
#real_x = real_x.reshape(-1,1)
#real_y = real_y.reshape(-1,1)

# now we have to split 70-30 or 80-20 training to test. import train_test_split
training_x, testing_x,training_y,testing_y = train_test_split(real_x, real_y, test_size=0.30,random_state=0) # 0.70 automatically goes to training

# NOw we have to build the model. import linear regression
LR = LinearRegression()
LR.fit(training_x, training_y)  # fit takes both training data


# training is complete now we have to predict
Pred_y = LR.predict(testing_x)

# y = b1x + b0
## 1.5 is years of experience. below is the predicted salary
print(LR.coef_) # prints the coefficient b1
print(LR.intercept_) # prints b0
print(9360.26128619 * 1.5 + 26777.3913412)
##

# print(testing_y[3]) # actual value
# print(Pred_y[3]) # prediction value

# part 2

# this is for training plot
plt.scatter(training_x, training_y, color ='red')
plt.plot(training_x, LR.predict(training_x), color='blue')
plt.title('Salary & Exp Training Plot')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
# red dot is actual values meaning this is actual salary given
# blue line is regression value meaning that is predicted salary

# below is for testing
plt.scatter(testing_x, testing_y, color ='red')
plt.plot(training_x, LR.predict(training_x), color='blue')
plt.title('Salary & Exp Testing Plot')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()