import pandas as pd
df=pd.read_csv("https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data")


columnas=["row.names","sbp","tobacco","ldl","adiposity","famhist","typea","obesity","alcohol","age","chd"]

df.columns=columnas
df.dtypes
#pruebo si tengo datos perdidos
df.isnull().sum()
#no tengo datos perdidos

#encontre una variable objeto famhist la paso a 1 0 con get_dummoes

dfhist_1=pd.get_dummies(df["famhist"])
dfhist_1

df=pd.concat([df,dfhist_1],axis=1)
df.head()
df=df.drop(["famhist"], axis=1)

#compruebo que ahora todos son numeros antes de normalizar
df.dtypes
a=len(columnas)
#vuelvo a transformar la lista columnas en columnas actuales
columnas=df.columns
#ahora normalizo las variables
for i in range(1,a):
    print(i)
    df[columnas[i]] =df[columnas[i]]/df[columnas[i]].max()

df.head()

#Ahora agrupo por metodo binning que en este caso no lo hare por que no es nescesario
#graficamos los datos
#edad y obesidad
df.plot(x="age",y="obesity",kind="scatter",figsize=(10,5))
#edad y tabaco
df.plot(x="age",y="tobacco",kind="scatter",figsize=(10,5))
#edad y alcohol
df.plot(x="age",y="alcohol",kind="scatter",figsize=(10,5))

##Analizamos machine learning###
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score

#ahora ya empezamos a modelar con ml, primero separo el tarjet del data

y=df["chd"]
X=df.drop("chd",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#Definir el algoritmo, elegi el modelo de suport vector machine me parece acertado porque no tengo muchos datos y puedo crear otra dimension, no gasta tantos recursos por ser pocos datos.
algoritmo = svm.SVC(kernel="linear")
#Entrenamos el modelo
algoritmo.fit(X_train,y_train)

#Realizamos la prediccion
y_test_pred =algoritmo.predict(X_test)

#ahora pasamos a las metricas primero matriz de confucion

print(confusion_matrix(y_test, y_test_pred))
#vemos que acerto re poco el modelo osea tenemos un acierto 71/93
71/93
#pasemos a las otras metricas
accuracy_score(y_test, y_test_pred)
precision_score(y_test, y_test_pred)
#tengo peor precision que exactitud

from sklearn.model_selection import cross_val_score
#como podemos preever les falta mas limpieza a nuestros datos pero la prediccion se encuentra en 70% podemos afirmar que una persona con diferentes rasgos somos capaz de predecir sus enfermedades hasta en un 70 %
