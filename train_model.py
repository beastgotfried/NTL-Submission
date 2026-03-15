#here we will try to implement linear regression with the help of pandas,numpy and matplotlib only without using the features preexisting in libraries such as sklearn
#using least squares method
#importing necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

def train_model(csv_file, plot=True):
    #importing the csv dataset as a pandas dataframe
    data=pd.read_csv('Salary_Data.csv')
    #now we need to find a relationship between YOE and salary to plot a graph #gathering infomartion of x and y

    x=data['YearsExperience'].values 
    y=data['Salary'].values

    #now we have collected the values of the experience and salary into our variables x and y #here they will store the data in one long array
    #so now we will calculate the mean of x and y for our linear math function using numpy
    mean_x=np.mean(x)
    mean_y=np.mean(y) 
    #this will store the values of mean of x and y as a float value

    #so now the equation of a line is y= mx+c where m is the slope and c is the intercept #to find m we will use the formula of the slope of the least squares of a regression line
    #finding number of input values i= len(x)

    #using the formula to calculate m and c
    numerator=0 
    denominator= 0 
    for i in range(len(x)):
        numerator += (x[i]-mean_x)*(y[i]-mean_y) 
        denominator +=(x[i]-mean_x)**2
        
    m=numerator/denominator #slope 
    c=mean_y-(mean_x*m)  #intercept

    #now plotting the graph since we have all the values for the line y=mx+c 
    # #for this we will be using matplotlib
    x_vals=np.linspace(np.min(x),np.max(x),100)  #finding min and max to plot the graph in x axis

    y_vals=m*x_vals+c  #implementing the logic to find y

    plt.plot(x_vals,y_vals,color='#0000FF',label='regression line') #plotting the line 
    plt.scatter(x,y,color='#FF0000',label='data') #plotting the points

    plt.xlabel('Years of Experience') #labelling the axis 
    plt.ylabel('Salary')
    plt.legend() 
    plt.show()
    return m,c
