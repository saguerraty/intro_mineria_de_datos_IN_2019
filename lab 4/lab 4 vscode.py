# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# ## Integrantes
# 
# 1. Gabriela Alfaro
# 2. Sebastian Guerraty
# 3. Maria Jose Jimenez
#%% [markdown]
# # Instrucciones
# 
# El laboratorio tiene 6 ptos, donde obtener 6 ptos equivale a un 7.0 y 0 ptos un 1.0. 
# 
# El formato de entrega será subir a u-cursos un Jupyter notebook
# laboratorio4.ipynb, que se debe ejecutar sin errores desde la primera celda a la última. Todo el código debe estar en el mismo notebook, el código debe estar comentado y testeado, el notebook debe estar escrito en forma de informe técnico, escribiendo una celda markdown antes o después de cada celda de código que arroja algún output. 
#%% [markdown]
# # Laboratorio 4: Clustering
# 
# Objetivos:
# 
# 
# 
# 1.  Entender en qué casos se puede utilizar clustering y cuál es su fin
# 2.  Conocer y aplicar modelos de clustering
# 3.  Conocer y aplicar métricas relacionadas a clustering
# 4. Entender diferencia entre clustering y aprendizaje no supervisado
#%% [markdown]
# # Investigación (2 ptos)
# 
# Elija 4 de 5 preguntas (`0.5 ptos c/u`)
# 
# 
# 
# 1. Explique un caso en que pueda fallar K-Means y mencione una forma de solucionarlo.
# 
# R: k-means al ser basado en la distancia euclidiana está orientado a clusters con forma circular, por lo que en clusters donde los grupos, o puntos pertenecientes a un cluster sean de una forma que se cruzen en términos de distancia euclidiana, k-means tendría dificultad para identificar correctamente los cluster apropiados.
# Una forma de corregir el problema es utilizar un método de clustering que permita identificar los cluster sin estar basado en distancia euclidiana, por ejemplo un método basado en densidad de puntos o vecinos más cercanos, como el caso del método DBSCAN
# <img src="images/kmeans_fail.png" style="width:600px;height:300px;">
# 2. ¿Es PCA un método de clustering? Justifique.
# 
# R: PCA es un método de compresión de información que es particularmente útil cuando la cantidad de atributos en un DF es grande, en particular cuando se tiene un número interesante de variables que indiquen un mismo concepto central. No es un método de clustering, ya que no asigna un identificador de pertenencia ni total ni parcial a algún grupo como si lo hace un método de clustering .
# 3. Investigue las siguientes métricas: *purity, silhouette score.* Describa ventajas y desventajas.
# 4. ¿En qué consiste el algoritmo Gaussian Mixture Models (GMM)? Comente su relación con K-Means.
# 
# R: Mixturas gausianas asume que los datos se componen de una cantidad N de distrubuciones gausianas de parámetros desconocidos, se puede considerar como una generalización de el método k-means ya que incorpora información de la coviarianza entre variables.
#    Adicionalmente, no tiene la limitación de k-means donde se orientan clusters con la distancia euclidiana, que termina en esferas en N dimensiones, GMM admite elipsoides debido a que asume que los datos dentro del cluster distribuyen como una gausiana.
# 5. Explique como hallaría el número "óptimo" de clusters en un problema de clustering.
# 
#%% [markdown]
# # Práctica (4 ptos)
# 
# 
# Considere el problema que enfrenta una empresa del retail que desea segmentar a sus clientes con el fin de entender mejor su comportamiento y así poder realizar ofertas específicas para cada grupo. 
# 
# Para lo anterior cuenta con los siguientes datos:
# 
# 
# 1.   Edad, género, educación, lugar dónde vive, teléfono, etc,
# 2.   Si es miembro o no del club de puntos, gastos realizados en un año, y una métrica otorgada (spending score) por el departamento de marketing que indica qué tan buenos gastadores son, donde 100 corresponde a lo más alto y 0 a lo más bajo. 
# 
# **Notas:**
# 
# 1.   No posee registro de los gastos de quienes no pertenecen al club de puntos, sin embargo según lo indicado por el departamento de marketing es una variable muy importante.
# 2.   A priori debería existir una correlación entre en el spending score y los gastos de una persona, aunque no necesariamente es así, por lo que se recomienda estudiar esta relación.
# 
#%% [markdown]
# **Tareas:**
# 
# 
# 1.   Realice un análisis exploratorio de los datos (cantidad de registros, medias, medianas, missing values, etc) y muestre al menos 2 gráficos de variables que considere relevantes para el análisis. (`0.5 ptos`)
# 2.   Cree una base de datos consistente (limpieza de NAs, transformaciones, imputaciones) y deje claramente expresadas las  * features* que utilizará para el clustering (al menos 3). Justifique las variables elegidas/creadas apoyándose en visualizaciones del punto 1.  (`0.5 ptos`)
# 3.    Utilice K-Means y con la ayuda del método del codo, encuentre el número "adecuado" de clusters, comente si lo encontrado por los métodos hace sentido y justifique su elección. Comente además respecto al tamaño de cada cluster y los centroides. (`1.5 ptos`)
# 
# `Nota:Se recomienda utilizar PCA y gráfico de radar con el fin de visualizar los clusters y sus centroides.`
# 
# 4. Utilice Clustering Jerárquico con al menos 2 linkage distintos al mostrado en clases y visualice sus respectivos dendogramas. Compare la cantidad de clusters encontrados con K-Means. ¿Se encuentra la misma cantidad? En caso de existir diferencias explique por qué cree que se dan. (`1.5 ptos`)
# 
# 
# 
#%% [markdown]
# # Importar Librerías

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import os


#%%
df=pd.read_csv('https://raw.githubusercontent.com/Camiloez/Labs-Data-Mining/master/data.csv')


#%%
df.head(5)

#%%
df.describe()

#%%
df.isna().sum()

#%%
df.dtypes
#%% [markdown]
# Matriz de correlacion para establecer relacion entre variables
corr = df.drop('CustomerID',axis=1).corr()
corr.style.background_gradient(cmap='coolwarm')

#%%
df['female'] = pd.get_dummies(df['Genre']).iloc[:,0]
sns.pairplot(df.iloc[:,5:13])


#%%
df["Expenses"]=df["Expenses"].replace('-',np.nan)
#%%
sns.scatterplot(x = "Annual Income (k$)",y= "Expenses", data= df[df["Expenses"].notnull()])

#%%
https://scikit-learn.org/stable/modules/mixture.html