#importing the same libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt    

#reading the csv datafile
df= pd.read_csv('titanic_data.csv')
#cleaning the data so that the learning algorithm doesnt get bugged since what i am cleaning are string based inputs which will break the algorithms
df=df.drop(["Name","Sex","Embarked"], axis=1)
df=df.dropna()
#cleaning more data because yes
x=df.drop("Survived",axis=1)
y=df["Survived"]
#same old assigning the models and splitting the data into training and validation
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)
# #calling the algorithm
# one cool thing to notice here is the max depth thing, this states how many subsequent branches will be created when the algorithm is called  
# this is really key in deciding how accurate the model will be and is actually one of the bgigest things that come into play when the model becomes overfitting  
# overfitting of the model is a problem since it is when the model starts to memorise the data instead of learning the pattern
classifier = DecisionTreeClassifier(max_depth=4,random_state=42)
classifier.fit(x_train,y_train)
#printing the accuracy of the algorithm
print(f"Accuracy: {classifier.score(x_test,y_test)}")
#decides the width and size of the graph
plt.figure(figsize=(15,10))
#same old plottig the  graph based on the parameters
plot_tree(classifier,feature_names=x.columns,class_names=["Not survived","Survived"],filled=True)
plt.show()