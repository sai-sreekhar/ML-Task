from matplotlib.animation import TimedAnimation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#LOAD DATA
titanic_data = pd.read_csv('titanic_train.csv')
len(titanic_data)
titanic_data.head()
titanic_data.index
titanic_data.columns
titanic_data.info()
titanic_data.dtypes
titanic_data.describe()

#DATA ANALYSIS
print(f"Max value of age column : {titanic_data['Age'].max()}")
print(f"Min value of age column : {titanic_data['Age'].min()}")

bins = [0, 5, 17, 25, 50, 80]
labels = ['Infant', 'Kid', 'Young', 'Adult', 'Old']
titanic_data['age'] = pd.cut(titanic_data['Age'], bins = bins, labels=labels)
titanic_data.head()

pd.DataFrame(titanic_data['age'].value_counts())
titanic_data['Age'].mode()[0]

bins = [-1, 7.9104, 14.4542, 31, 512.330]
labels = ['low', 'medium-low', 'medium', 'high']
titanic_data['fare'] = pd.cut(titanic_data["fare"], bins = bins, labels = labels)

titanic_data['Embarked'].unique()
pd.DataFrame(titanic_data['Embarked'].value_counts())



sns.countplot(x="Survived",data=titanic_data) #countplot of subrvived vs not  survived
sns.distplot(titanic_data['Survived'])
sns.countplot(x='Survived',data=titanic_data,hue='Sex') ##Male vs Female Survived?
sns.displot(x='Age',data=titanic_data) ##find the distribution for the age column
plt.show()

plt.scatter(titanic_data.Age,titanic_data.Survived,marker = '+',color ='red')
titanic_data[['Pclass', 'Survived']].groupby(['Pclass']).sum().sort_values(by='Survived')
titanic_data[['Sex', 'Survived']].groupby(['Sex']).sum().sort_values(by='Survived')

plt.figure(figsize=(20, 10))
plt.subplot(321)
sns.barplot(x = 'SibSp', y = 'Survived', data = titanic_data)
plt.subplot(322)
sns.barplot(x = 'fare', y = 'Survived', data = titanic_data)
plt.subplot(323)
sns.barplot(x = 'Pclass', y = 'Survived', data = titanic_data)
plt.subplot(324)
sns.barplot(x = 'age', y = 'Survived', data = titanic_data)
plt.subplot(325)
sns.barplot(x = 'Sex', y = 'Survived', data = titanic_data)
plt.subplot(326)
sns.barplot(x = 'Embarked', y = 'Survived', data = titanic_data)

titanic_data.isna() #Check for null
titanic_data.isna().sum() #Check how many values are null
sns.heatmap(titanic_data.isna()) #Visualize null values

(titanic_data['Age'].isna().sum()/len(titanic_data['Age']))*100 #find the % of null values in age column
(titanic_data['Cabin'].isna().sum()/len(titanic_data['Cabin']))*100 #find the % of null values in cabin column

#DATA CLEANING
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True) #fill age column
titanic_data['Age'].isna().sum() #verify null value
sns.heatmap(titanic_data.isna()) #verify null value 
plt.show()

titanic_data.drop("Cabin",axis=1,inplace=True) #Drop cabin column
titanic_data.head() #see the contents of the data


titanic_data.info() #Check for the non-numeric column
titanic_data.dtypes

gender=pd.get_dummies(titanic_data['Sex'],drop_first=True) #convert sex column to numerical values
titanic_data['Gender']=gender 

titanic_data.head()
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)

dummies = ['Embarked']
dummy_data = pd.get_dummies(titanic_data[dummies])
dummy_data.shape

titanic_data = pd.concat([titanic_data, dummy_data], axis = 1)
titanic_data.drop(dummies, axis=1, inplace=True)
titanic_data.columns

titanic_data.drop(['Name','Sex','Ticket','age','fare'],axis=1,inplace=True)
titanic_data.head()

#Seperate Dependent and Independent variables
x=titanic_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender','Embarked_C','Embarked_Q','Embarked_S']]
y=titanic_data['Survived']

#DATA MODELLING
from sklearn.model_selection import train_test_split #import train test split method
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42) #train test split

from sklearn.linear_model import LogisticRegression #import Logistic  Regression
lr=LogisticRegression() #Fit  Logistic Regression 
lr.fit(x_train,y_train)
predict=lr.predict(x_test) #predict

#TESTING
from sklearn.metrics import confusion_matrix ,accuracy_score#print confusion matrix 
pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])
from sklearn.metrics import classification_report #import classification report
print(classification_report(y_test,predict))
accuracy_score(predict, y_test)




















