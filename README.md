# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mohamed Anas O.I
RegisterNumber:  212223110028
*/
```

```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee.csv")
data.head()
```
### Output:
![Screenshot 2024-10-16 084015](https://github.com/user-attachments/assets/ca96e358-32bd-422f-af6f-e21c24d23329)

```
data.info()
data.isnull().sum()
data["left"].value_counts()
```
### Output:
![Screenshot 2024-10-16 084021](https://github.com/user-attachments/assets/3f530580-9103-43bb-8444-8a5cce0abe54)


```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
### Output:
![Screenshot 2024-10-16 084029](https://github.com/user-attachments/assets/f8f74856-dd62-40ab-a00b-f0b741916021)

```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
### Output:
![Screenshot 2024-10-16 084036](https://github.com/user-attachments/assets/c9c02fcb-40da-49e4-ab71-471f30c67b99)

```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
### Output:
![Screenshot 2024-10-16 084041](https://github.com/user-attachments/assets/e163c234-f48b-4ef3-980d-6756379f8141)

```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
### Output:
![Screenshot 2024-10-16 084058](https://github.com/user-attachments/assets/9d5d48e4-4aa6-4a24-b124-bcec4e161646)

```
import matplotlib.pyplot as plt
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
### Output:
![Screenshot 2024-10-16 084115](https://github.com/user-attachments/assets/26f31402-8f35-4eee-93a6-791254b17e94)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
