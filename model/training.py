import numpy as np
import pandas as pd
import joblib as jb
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_csv('./Training.csv')
X = df.iloc[:, 0:132]
Y = df['prognosis']
le = LabelEncoder()
Y = le.fit_transform(Y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Predict and evaluate
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# Save model if good enough
if accuracy >= 0.93:
    print(f"The accuracy of the model is {accuracy}")
    jb.dump(model, 'diseasepred.pkl')
    print("Model saved successfully")
else:
    print(f"The accuracy of the model is {accuracy}")
    print("Model not saved")
