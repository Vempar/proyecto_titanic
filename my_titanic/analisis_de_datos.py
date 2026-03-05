import pandas as pd
import matplotlib.pyplot as plt

train= pd.read_csv("my_titanic/datasets/train.csv") 
test= pd.read_csv("my_titanic/datasets/test.csv") 

print(train.columns)
print(train.shape)
print (train.info())
print (train.describe())
#describe que nos indica como se comportan las variables y ver valore unicos
print (train.describe(include='object'))

#ver los valores unicos de una variable
print (train['Sex'].unique())
#ver los valores unicos de una variable
print (train['Sex'].value_counts())

#ver los valores unicos de una variable
print (train['Sex'].value_counts(normalize=True))

#saber los pasajeros que sobrevivieron
#print (train[train['Survived']==1])

#saber los pasajeros que no sobrevivieron
#print (train[train['Survived']==0])

#media de los pasajeros que sobrevivieron por sexo
#train.groupby('Sex')['Survived'].mean()
#cantidad de pasajeros que sobrevivieron por sexo
print(train.groupby(['Survived']).count()['PassengerId'])
#cantidad de pasajeros que sobrevivieron por sexo
group=train.groupby(['Survived','Sex']).count()['PassengerId']
print(group)

#cantidad de pasajeros que sobrevivieron por sexo y clase
group_pclass=train.groupby(['Survived','Sex','Pclass']).count()['PassengerId']
print(group_pclass)

#cantidad de pasajeros que sobrevivieron por sexo y clase y edad
group_pclass_age=train.groupby(['Survived','Sex','Pclass','Age']).count()['PassengerId']
print(group_pclass_age)

print(group.unstack(level=0))
print(group.unstack(level=0).index)
print(group.unstack(level=0).values)

#grafico de barras de los pasajeros que sobrevivieron por sexo
plt.figure(figsize=(10,6))
(group.unstack(level=0).plot(kind='bar'))
plt.show()

#grafico de barras de los pasajeros que sobrevivieron por sexo y clase
plt.figure(figsize=(10,6))
(group_pclass.unstack(level=0).plot(kind='bar'))
plt.show()
#grafico de barras de los pasajeros que sobrevivieron por sexo y clase y puerto de embarque
plt.figure(figsize=(10,6))
(train.groupby(['Pclass','Embarked','Survived']).count()['PassengerId'].unstack(level=0).plot(kind='bar'))
plt.show()

