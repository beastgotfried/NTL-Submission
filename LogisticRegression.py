import pandas as pd   #read the csv and dataset
    
from sklearn.linear_model import LogisticRegression  #importing the algorithm
from sklearn.model_selection import train_test_split  #importing to split the data
from sklearn.preprocessing import StandardScaler    #import feature, will talk later    
from sklearn.metrics import accuracy_score,classification_report  #import output values


df = pd.read_csv('titanic_data.csv') # #importing the data 
df=df.drop(["Name","Sex","Embarked"], axis=1) #cleaning the data since this has values i cannot use
df=df.dropna() #logistic regression cannotaccept nan values, hence it dropped every row that had data that did not match its criterias


x= df.drop("Survived", axis= 1)   #assigning x to everything apart from survived column
y= df["Survived"] #assigning y the survived column which is the output column giving values either 0 to 1

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42) #training the model, random_State=42 makes sure the model takes data from the given dataset to check for output to mantain uniformity

scaler= StandardScaler() #importing standardscaler function to make sure each criteria/parameter judges the model equally. 
#in logistic regression often
x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(max_iter=500)

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

print(f"Accuracy: {accuracy_score(y_test,y_pred)}")
print(f"Classification report:  {classification_report(y_test,y_pred)}")

