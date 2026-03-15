**Linear Regression with and without Sklearn**


=> So there are 2 main components of this project:

***FEATURE 1***
1) The first feature of the project is using the least squares method to find and plot the LinearRegression graph for a sample dataset that i took from kaggle regarding the salary data of people compared to their years of experience
2) this part revolves around using the libraries numpy,pandas and matplotlib in their own functional domains to form the final product
3) numpy is used for computation of the mean,finding the minimum and maximum along with the math formulas computation
4) the math formulas used in this were for the r^2 values and formulas used to find the values of m and c which was done with the help of data value that was gathered using pandas after reading the data in the csv file
5) matplotlib was used to plot the data in the function that was gathered and the line graph of regression was plotted along with the data points


**How did i check the accuracy of the model?!?**
I have used 2 major methods to check the accuracy of the model,
#Method 1
1) the first method i have used to check the accuracy revolved around defining a function in the main file which trained the model
2) then i split the data set into 80-20 and used to the 20% to validate the data while running the model on the entire training data
3) after splitting the data i ran the model and printed the output through a panda table

#Method 2 
1) the second method that i used was finding the R^2 value for the table
2) The R^2 value tells us about the variance of the data with the predicted data by the model
3) the formula of r^2 is derived around by taking the complement of the difference of the actual value subtracted by the predicted value divided by the actual value subtracted by the mean of the predicted values
4) came out to be around 0.75 in this data, which is around 75% accuracy
 COOL!! PRETTY GOOD FOR THE SIZE OF THE DATA WE HAD!! (only 30 rows)

***FEATURE 2***
1) The second feature of the project lies around using the sklearn library
2) This Library already has a predefined function for linear regression computation
3) This made finding the values of m and c easy even though it was a function with just 1 feature
4) AfAfter finding the values of m and c the linear regression line was plotted completely and tthe data points were marked and compared to the original data

<img src=comparing_model.png" alt="Comparing both outputs">