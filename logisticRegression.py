import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

data = pd.read_csv('resources/ads.csv')

# we have to choose the data
real_x = data.iloc[:, [2,3]].values
real_y = data.iloc[:, 4].values

# now we have to split the data
training_x, testing_x, training_y, testing_y = train_test_split(real_x, real_y, test_size=0.25, random_state=0)

# here age and salary is independent variables
# here the difference (data gap) between age is salary is high, in that case we do 'feature Scaling' and we import

scaler = StandardScaler() # its changes data within the range -2 t0 2
training_x = scaler.fit_transform(training_x)

testing_x = scaler.fit_transform(testing_x)

print(training_x)
print(testing_x)

classifier_LR = LogisticRegression(random_state=0)
classifier_LR.fit(training_x, training_y)

y_pred = classifier_LR.predict(testing_x)
print(y_pred)
print(testing_y)

# now we need to import confusion matrix because we cant compute one-by-one

conf_mat = confusion_matrix(testing_y, y_pred)
print(conf_mat)
# look at the diagonals sum it up, it means 63 + 24 is correct observation i.e. 87 % correct, 8+5 i.e. 13 % incorrect observation

# this is different to graph because we used feature scaling since the data gap between age and salary was high,
# so we import ListedColormap

x_set, y_set = training_x, training_y

x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max()+1, step = 0.01),
                     np.arange(start = x_set[:,1].min() - 1, stop = x_set[:,1].max()+1,step = 0.01))
plt.contourf(x1,x2, classifier_LR.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape), alpha= 0.75, cmap=ListedColormap(('red','green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# this is training data
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j ,1],
                c = ListedColormap(('red','green'))(i), label = j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# this to plot testing data
x_set, y_set = testing_x, testing_y

x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max()+1, step = 0.01),
                     np.arange(start = x_set[:,1].min() - 1, stop = x_set[:,1].max()+1,step = 0.01))
plt.contourf(x1,x2, classifier_LR.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape), alpha= 0.75, cmap=ListedColormap(('red','green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# this is training data
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j ,1],
                c = ListedColormap(('red','green'))(i), label = j)

plt.title('Logistic Regression (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()












