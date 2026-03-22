import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt    

df= pd.read_csv('titanic_data.csv')

df=df.drop(["Name","Sex","Embarked"], axis=1)
df=df.dropna()

x=df.drop("Survived",axis=1)
y=df["Survived"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

classifier = DecisionTreeClassifier(max_depth=4,random_state=42)
classifier.fit(x_train,y_train)

print(f"Accuracy: {classifier.score(x_test,y_test)}")

plt.figure(figsize=(15,10))
plot_tree(classifier,feature_names=x.columns,class_names=["Not survived","Survived"],filled=True)
plt.show()