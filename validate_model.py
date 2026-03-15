#now we will be validating the model that we have built in the train_model file

import pandas as pd
import numpy as np

from train_model import train_model   #importing our training function
#GETTING THE VALues of m and c from the original function since they wont change even if we add or subtract few values from the dataset for validation
m,c = train_model('Salary_Data.csv',plot=False)

data = pd.read_csv('Salary_Data.csv')
x=data['YearsExperience'].values
y=data['Salary'].values
 
#splitting validation data to be 20% of the entire data
split = int(0.8*len(x))
#forming an array with the values of data from  csv
x_test,y_test = x[split:], y[split:]
#running the function on the values
y_pred= m*x_test+c
#printing validation data along with original data
results = pd.DataFrame({
    "YearsExperience": x_test,
    "Actual Salary": y_test,
    "Predicted Salary": np.round(y_pred,2),
    "Error": np.round((y_test-y_pred),2),
    "Error ": np.round((1-(y_pred/y_test)),2)
})

print(results)

#NOW THIS WAS JUST ONE WAY ON HOW WE CAN PROVE THE MODELS WORTH

#OTHER METHODS:
#one method apart from this is finding the r^2 value which will tell us the coefficient of regression 
# which in simpler terms is just the variance of the dataset points from the original formed regression line

#FINDING r^2
ss_res = np.sum((y_test-y_pred)**2)
ss_tot = np.sum((y_test-np.mean(y_test))**2)

r = 1- (ss_res/ss_tot)

print(r)
#this will give us the rating of the model, in this case with the data i have im getting a value of approx 75% which means the model functions at 75% accuracy
