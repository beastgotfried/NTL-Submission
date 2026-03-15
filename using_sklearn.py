import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#loading the dataset that we have of the salary data

data = pd.read_csv('Salary_Data.csv')

x=data[['YearsExperience']].values #2D array for extracting the years of experience in the form (num,1)
y=data['Salary'].values #1D array for extracting the salary

#creating the model that we have

model = LinearRegression() #pulling the linear regression model from sklearn
model.fit(x,y)  #making the model fit for the x,y values

#getting the values of m and c for the model to plot the line 

m= model.coef_[0] #pull the coefficent out at position 0, we only have 1 coefficient here since we only have one feature in the dataset which is the YOE
c= model.intercept_ #finding the intercept wiht predefined function

y_pred= model.predict(x) #finding salary by predicciting the model for x

#plotting with matplotlib
plt.plot(x,y_pred, color= '#0000FF', label = "Regression Line")
plt.scatter(x,y, color = '#FF0000' , label = "Data Points")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()