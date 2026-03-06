import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train= pd.read_csv("my_titanic/datasets/train.csv") 
test= pd.read_csv("my_titanic/datasets/test.csv") 

train2=train[['Survived','Sex','Pclass','Age','PassengerId']]
print (train2.info())

#ver los pasajeros que no tienen edad conocida por sexo y clase
print (train2[train2['Age'].isna()]
        .groupby(['Sex','Pclass'])
        .count()
        .unstack(level=0))
#ver los pasajeros que no tienen edad por hermanos y padres o solos
print (train[train['Age'].isnull()]
        .groupby(['SibSp','Parch'])
        .count()['PassengerId']
        .unstack(level=0))

#calcular la edad media por sexo y clase
print (train2['Age'].mean())
#calcular la mediana de age
print (train2['Age'].median())
#calcular la edad media por sexo y clase
print (train.groupby(['Sex','Pclass'])['Age'].mean())
#calcular la edad media por sexo y clase y hermanos y padres
print (train.groupby(['Sex','Pclass','SibSp','Parch'])['Age'].mean())

#se establece una eda media de 23 años para los pasajeros que no tienen edad conocida se establece esta edad por los varones del grupo sin hijos
train2['Age']=train2['Age'].fillna(23)
print (train2.info())

#map para transformar variables categoricas en numericas y lo pasamos a int
train2['Sex']=train2['Sex'].map({'female':0,'male':1}).astype(int)
print (train2.info())
print (train2.head())

#crea una nueva columna en la que indica si el pasajero  esta solo  y sin familia
train2['Alone']=np.where(((train['SibSp']==0)&(train['Parch'])==0),1,0)
print (train2.info())
print (train2.head())

Group_alone= train2.groupby(['Survived','Alone']).count()['PassengerId']
print (Group_alone)

(Group_alone.unstack(level=0).plot(kind='bar'))
plt.show()

print (train2.info())
train3=train2[['Survived','Sex','Age','Pclass','Alone']]
#variable dependiente
y=train3['Survived']
#variables independientes
x=train3[['Sex','Age','Pclass','Alone']]
print (y.shape, x.shape)

#eliminar la columna embarked
#train2=train2.drop('Embarked',axis=1)
#print (train2.info())
