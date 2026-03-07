import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay

test= pd.read_csv("my_titanic/datasets/test.csv") 
train= pd.read_csv("my_titanic/datasets/train.csv") 
train2=train[['Survived','Sex','Pclass','Age','PassengerId']]

#variables para generar el nombre del archivo
name_tree="fichero_modelo_tree"
name_log="fichero_modelo_log"
Complemento_archivo=np.random.randint(10000000, 99999999)

#modelo de entrenamiento
#se establece la edad media para los pasajeros que no tienen edad conocida se establece esta edad por los varones del grupo sin hijos
train2['Age']=train2['Age'].fillna(train2['Age'].mean())

#map para transformar variables categoricas en numericas y lo pasamos a int
train2['Sex']=train2['Sex'].map({'female':0,'male':1}).astype(int)

#crea una nueva columna en la que indica si el pasajero  esta solo  y sin familia
train2['Alone']=np.where(((train['SibSp']==0)&(train['Parch'])==0),1,0)

train3=train2[['Survived','Sex','Age','Pclass','Alone']]
#variable dependiente
y=train3['Survived']
#variables independientes
x=train3[['Sex','Age','Pclass','Alone']]

test2=test[['Sex','Pclass','Age','PassengerId']]

#modelo final para test
#se establece la edad media para los pasajeros que no tienen edad conocida se establece esta edad por los varones del grupo sin hijos
test2['Age']=test2['Age'].fillna(test2['Age'].mean())

#map para transformar variables categoricas en numericas y lo pasamos a int
test2['Sex']=test2['Sex'].map({'female':0,'male':1}).astype(int)

#crea una nueva columna en la que indica si el pasajero  esta solo  y sin familia
test2['Alone']=np.where(((test['SibSp']==0)&(test['Parch'])==0),1,0)

#entrenamiento del modelo de arbol de decision
decisiontree=DecisionTreeClassifier()
decisiontree.fit(x,y)
#entrenamiento de los modelos de regresion logistica 
logreg=LogisticRegression()
logreg.fit(x,y)

features=['Sex','Age','Pclass','Alone']
x_text= test2[features]

Y_pred_tree= decisiontree.predict(x_text)
print(Y_pred_tree[0:10])
Y_pred_logreg= logreg.predict(x_text)
print(Y_pred_logreg[0:10])



def download_predec_tree (Y_pred, name):
    output = pd.DataFrame({'PassengerId': test['PassengerId'],
                            'Survived': Y_pred })
    output.to_csv(name, index=False)
    print ("El archivo se ha guardado como " + name_tree)    

download_predec_tree(Y_pred_tree, 'my_titanic/datasets/'+name_tree+"_"+str(Complemento_archivo)+".csv")


def download_predec_log (Y_predL, name):
    output = pd.DataFrame({'PassengerId': test['PassengerId'],
                            'Survived': Y_predL })
    output.to_csv(name, index=False)
    print ("El archivo se ha guardado como " + name_log)    

download_predec_log(Y_pred_logreg, 'my_titanic/datasets/'+name_log+"_"+str(Complemento_archivo)+".csv")