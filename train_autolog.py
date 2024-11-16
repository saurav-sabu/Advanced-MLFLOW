import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


df =  pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")

X = df.drop("species",axis=1)
y = df["species"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

max_depth = 1
n_estimators = 100

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.autolog()

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test,y_pred)

    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    
    
    #  mlflow.log_artifact(__file__)
    
    mlflow.set_tag("author","saurav")
    mlflow.set_tag("model_name","Random Forest")
