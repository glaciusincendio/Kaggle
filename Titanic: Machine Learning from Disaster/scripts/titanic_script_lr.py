#libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#paths
train_data_path = "*/Kaggle/Titanic_Machine_Learning_from_Disaster/data/train.csv"
test_data_path = "*/Kaggle/Titanic_Machine_Learning_from_Disaster/data/test.csv"
target_data_path = "*/Kaggle/Titanic_Machine_Learning_from_Disaster/"

#variables
test_size = 0.1
random_seed = 79

#read train data
train_data = pd.read_csv(train_data_path)

#seperating features and labels
X = train_data.iloc[:,2:]
Y = train_data['Survived']

#dropping columns
X = X.drop(['Name'],axis=1)
X = X.drop(['Ticket'],axis=1)
X = X.drop(['Fare'],axis=1)
X = X.drop(['Cabin'],axis=1)
X = X.drop(['Embarked'],axis=1)

#filling age column with mean values inplace of missing values
X['Age'].fillna(X.Age.mean(),inplace=True)

#Converting gender to one hot
Gender = pd.get_dummies(X['Sex'])

#dropping converted column
X = X.drop(['Sex'],axis = 1)

#concatenate one hot column 
X = pd.concat((X, Gender), axis=1)

#Spltting into test & validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=random_seed, stratify=Y)

# status update
print("Train Data Shape: ", X_train.shape, Y_train.shape)
print("Validation Data Shape: ", X_val.shape, Y_val.shape)

# getting model object
model = LogisticRegression()

# training model
model.fit(X_train, Y_train)

# validating model
pred = model.predict(X_val)

#getting accuracy
acc = accuracy_score(pred,Y_val)


#read test data
test_data = pd.read_csv(test_data_path)

#seperating features and labels
X_test = test_data.iloc[:,1:]

#dropping columns
X_test = X_test.drop(['Name'],axis=1)
X_test = X_test.drop(['Ticket'],axis=1)
X_test = X_test.drop(['Fare'],axis=1)
X_test = X_test.drop(['Cabin'],axis=1)
X_test = X_test.drop(['Embarked'],axis=1)

#filling age column with mean values inplace of missing values
X_test['Age'].fillna(X_test.Age.mean(),inplace=True)

#Converting gender to one hot
Gender = pd.get_dummies(X_test['Sex'])

#dropping converted column
X_test = X_test.drop(['Sex'],axis = 1)

#concatenate one hot column 
X_test = pd.concat((X_test, Gender), axis=1)

#predicting on test data
pred = model.predict(X_test)

#creating submission file
df = pd.DataFrame()
PassengerId = []
Survived = []

#converting to dataframe
PassengerId = test_data['PassengerId']
Survived = pred

df = pd.DataFrame(list(zip(PassengerId, Survived)), columns =['PassengerId', 'Survived'])

#creating csv file
df.to_csv(target_data_path+ 'results/' + "submission_lr.csv", index=False)

# save the model to disk
filename = 'finalized_model_lr.sav'
pickle.dump(model, open(target_data_path + 'model/' + filename, 'wb'))

#On submission of Logistic Regression model the score was 0.75598 on 05/11/2019
