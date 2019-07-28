import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#HOW TO IMPORT DATA SET:
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#SPLITTING THE DATASET INTO TEST SET AND TRAINING SET:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

"""#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""


#FITTING SIMPLE LINEAR REGRESSION ALGORITHM IN THE TRAINING SET:
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(x_train, y_train)


#PREDICTIMG THE TEST SET RESULTS:
y_pred = Regressor.predict(x_test)


#VISULISING THE TRAINING SET RESULTS:
plt.scatter(x_train, y_train, color = 'red')  #OBSERVATION POINTS
plt.plot(x_train, Regressor.predict(x_train), color = 'blue') #PROJECTING REGRESSION LINE
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#VISULISING THE TEST SET RESULTS:
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, Regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(Regressor.coef_)
print(Regressor.intercept_)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)