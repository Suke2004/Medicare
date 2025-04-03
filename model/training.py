import numpy as np
import pandas as pd
import joblib as jb
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df=pd.read_csv(r'model\Training.csv')
X=df.iloc[:,0:132]
Y=df['prognosis']
le=LabelEncoder()
Y=le.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.6,random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
max=0;
for i in range(1,12):
    accuracy=accuracy_score(Y_pred,Y_test)
    if(accuracy>=max):
        max=accuracy
if(max>=0.93):
    print("The accuracy of the model is ",max)
    jb.dump(model,'diseasepred.pkl')
    print("Model saved succesfully")
else:
    print("The accuracy of the model is ",max)
    print("Model not saved")