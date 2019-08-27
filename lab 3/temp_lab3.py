# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
try:
	os.chdir(os.path.join(os.getcwd(), 'lab 3'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Integrantes
# 
# 1. Sebastian Guerraty
# 2. María José Jiménez
# 3. Gabriela Alfaro
#%% [markdown]
# # Instrucciones
# 
# El laboratorio tiene 6 ptos, donde obtener 6 ptos equivale a un 7.0 y 0 ptos un 1.0. 
# 
# El formato de entrega será subir a u-cursos un Jupyter notebook
# laboratorio1.ipynb, que se debe ejecutar sin errores desde la primera celda a la última. Todo el código debe estar en el mismo notebook, el código debe estar comentado y testeado, el notebook debe estar escrito en forma de informe técnico, escribiendo una celda markdown antes o después de cada celda de código que arroja algún output. 
#%% [markdown]
# ## 1. Teórico, 2 ptos 
# 
# 1.1 ¿En qué consisten las transformaciones Box-Cox, Z-Score y
# Max-Min? Describa al menos 2 ventajas y desventajas de cada una de
# estas transformaciones en una tabla comparativa. **(0.5 ptos)**
# 
# 1.2 ¿Es estrictamente necesario realizar una transformación de los
# atributos en un proceso de minería de datos? ¿En qué casos
# podría no ser necesario? De al menos un ejemplo. **(0.5 ptos)**
# 
# 1.3 Suponga que está analizando una encuesta con datos socioeconómicos (similar al Censo), y descubre que la variable **sueldo** tiene un 60% de datos perdidos. A qué tipo de dato perdido cree que corresponde: ¿MCAR, MAR, NMAR? Justique su elección ¿Qué haría con esta variable suponiendo que su objetivo es predecir nivel de estudios de una persona (enseñanza básica, media, universitaria, etc)? Explique. **(1.0 pto)**
# 
# 
# ## 2. Aplicación, 4 ptos
# 
# ---
# Considere el problema que enfrenta una entidad financiera que tiene altas tasas de fuga voluntaria. Esta institución no tiene claro cuál es el perfil característico que tienen los clientes fugitivos ni cuáles son las razones por las cuales estos se fugan.
# 
# El gerente general le ha pedido **definir el patrón característico de los clientes fugitivos y de los clientes no fugitivos con el objetivo de definir una serie de políticas comerciales** que permitan retener a estos potenciales clientes fugitivos.
#%% [markdown]
# **Tareas:**
# 
# 2.1 Resuelva los problemas de inconsistencia y valores pérdidos en la base de datos, justificando cada una de sus decisiones. Impute al menos una variable utilizando regresión lineal con el resto de columnas como regresores. 
# **Nota**: no deben quedar valores perdidos. **(1.5 ptos)**
# 
# 
# 2.2 Aplique 2 métodos de transformación sobre 2 variables distintas, grafique la distribución antes y después. Comente si las transformaciones tienen sentido y cuál sería su utilidad **(1.0 pto)**
# 
# 
# 2.3 Aplique las técnicas de selección que considere pertinentes y elimine de la base aquellos atributos que no son relevantes según sus criterios, elimine al menos 2 **(1.0 pto)**
# 
# 
# 2.4 Comente respecto a patrones encontrados entre clientes fugitivos y no fugitivos. Apóyese en gráficos y test estadísticos. **(0.5 ptos)**
# 
# 
# ---
# Variable | Descripción
# ------------- | -------------
# 1. customer | ID
# 2. Age | Edad
# 3. Employ | Años en el mismo empleo
# 4. Address | Años viviendo en el mismo lugar
# 5. Income | Ingreso en USD
# 6. Debtinc | Ratio Ingreso/Deuda
# 7. Creddebt | Monto de deuda en tarjetas de crédito
# 8. OthDebt | Monto de otras deudas
# 9. Education | Nivel educacional
# 10. Nationality | Nacionalidad
# 11. Default | Variable objetivo

#%%
import pandas as pd


#%%
url = "https://raw.githubusercontent.com/Camiloez/lab3-dataset/master/data_lab.csv"
df = pd.read_csv(url)


#%%
df.head(5)
#df[df['Age']>=80]
#df[df['Income']<0]
#df['Default'].unique()
#df['Education'].unique()


#%%
df.describe()

#%%
df.dtypes
#%%
fig, df_plot=plt.subplots(nrows=1,ncols=4)
df_plot[0].hist(df['Age'], range=[17,90])
df_plot[0].set_title('Age')
df_plot[1].hist(df['Address'])
df_plot[1].set_title('Address')
df_plot[2].hist(df['Income'], range=[0,600])
df_plot[2].set_title('Income')
df_plot[3].hist(df['Debtinc'])
df_plot[3].set_title('Debtinc')
df_plot

df['Education'] = df['Education'].astype(str)
modi_df=df.copy()
modi_df['Education']=str(modi_df['Education'])
modi_df['Default']=str(modi_df['Default'])

#%%
pd.Series(df['Education']).value_counts().plot('bar')
#%%
df.isna().any(axis = 0)
#%%
moda_edad =  float(df['Age'].mode())
modi_df.loc[modi_df['Age']>100 , 'Age'] = np.nan
modi_df['Age'].fillna(value = moda_edad, inplace=True)
moda_adress = float(df['Address'].mode())
modi_df['Address'].fillna(value=moda_adress,inplace=True)

#%%
linreg = LinearRegression()
modi_income_temp = modi_df.copy()
modi_income_temp = modi_income_temp.drop(columns=['Default','Education','Nationality'])
modi_temp_x = modi_income_temp[modi_income_temp['Income'].notnull()].drop(columns="Income", axis=1)
modi_temp_y = modi_income_temp[modi_income_temp['Income'].notnull()]

modi_test = modi_income_temp[modi_income_temp['Income'].isnull()].drop("Income", axis=1)
#%%
modi_temp_x = modi_temp_x.astype(float).fillna(0.0)
modi_test = modi_test.astype(float, errors='ignore').fillna(0.0)
modi_temp_y = modi_temp_y.astype(float).fillna(0.0)
#%%
linreg.fit(modi_temp_x,modi_temp_y)
predicted = linreg.predict(modi_test)
modi_df.Income[modi_df.Income.isnull()] = predicted
#%%
modi_df['Debtinc'].interpolate(method='linear', inplace= True)
#%% [markdown]
#Las transformaciones a utiliar va a ser tomar el logaritmo

#%%
fig, df_mod=plt.subplots(nrows=1,ncols=2)
df_mod[0].hist(modi_df['Income'])
df_mod[0].set_title('Income')
df_mod[1].hist(modi_df[''])


#%%
modi_df['Income']= np.log(modi_df['Income'])

#%%
fig, df_after=plt.subplot(nrow=1,ncols=2)
df_after[0].hist(modi_df['Income'])
df_after[0].set_title('Income')