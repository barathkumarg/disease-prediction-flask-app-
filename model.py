import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

#heart disease dataset
df=pd.read_csv("heart.csv")
#print(df)
x=df.drop(['target'],axis=1)
y=df[['target']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
pred=model.predict(x_test)
#print(pred)

pickle.dump(model, open('model_heart_disease.pkl','wb'))

#**************************************************************
#diabetics data set
pdata = pd.read_csv("pima-indians-diabetes.csv")
X = pdata.drop('class',axis=1)     # Predictor feature columns (8 X m)
Y = pdata['class']   # Predicted class (1=True, 0=False) (1 X m)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
model.fit(X_train, Y_train)
Pred=model.predict(X_test)


pickle.dump(model, open('model_diabetics.pkl','wb'))