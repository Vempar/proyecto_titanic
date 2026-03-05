import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

train= pd.read_csv("my_titanic/datasets/train.csv") 
test= pd.read_csv("my_titanic/datasets/test.csv") 

#entrenamiento de los modelos de regresion logistica 
logreg=LogisticRegression()

#entrenamiento del modelo de arbol de decision
decisiontree=DecisionTreeClassifier()

print (train.info())
print (test.info())
