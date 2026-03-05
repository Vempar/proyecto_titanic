import pandas as pd

train= pd.read_csv("my_titanic/datasets/train.csv") 
test= pd.read_csv("my_titanic/datasets/test.csv") 

train2=train[['Survived','Sex','Pclass','Age']]
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


