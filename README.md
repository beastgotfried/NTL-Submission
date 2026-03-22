***TASK-3 DIFFERENTIATING Logistic Regression and Decision Trees***

![alt text](images\logistic-output.png)
![alt text](images\decision-tree-output.png)

So in this branch of the repository i have pushed the code for differentiating between LR and DT and now i will be explaining my outcomes based on my analysis and what i learnt throughout the process of executing the task

**Logistic Regression**
Logistic regression is a parametric based classification algorithm that works on classifying whether a situation will occur or not
this works by using a simple sigmoid function that is used to compute the occurence of a scenario and play around with the plotted graph that way

what i observed while playing with my data is that logistic regression is good at classifying values when there are a msall number of parameters. it is both linear and non linear in a nature

linear because of how it takes input into the function since the features are all single step functions and non linear in the aspect that it used  a sigmioid function which has curvy nature graph

it uses probability to predict the values of a situation happening and hprepare a pattern based on it


**Decision Trees**
Decision trees are different, they are less math based and more pattern and analysis based. Decision trees work on simple yes and no criterias where every feature becomes a seperate branch based on yes and no features. This does not mean that th e number of branches of my system will be equal to the number of parameters i have. the parameters can be grouped into general cases as well

We can decide the number of branches we want to create and make the model perfectly fit by playing around with those values. 
The math behind them is simple 0 or 1 values in case of binary classification which i have learnt and implemented till now, lets see how it goes in the future


***THE DIFFERENCE***
Here in case of me implementing my dataset which gave a result of whether the indivual will survive or not
both algorithms gave roughly the accuracy of 70%, this was because both the algorithm function one way or the other even though one model was parametric and the other was non parametric

the logistic regression method is partially linear and partially non linear making it easier to tune the model to finer values.
the model could not achieve overall high accuracy due to relatively small dataset due to pruning and cleaning of the data as per our requirement of the algorithm

decision trees were more rule based non linear models which functioned on trees instead of graph hence had their own visualisation pattern

the models gave roughly the same accuracy and the overall accuracy of the model could be increased by increasing the dataset and providing more accurate features to judge for the model