# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'lab 2'))
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
# laboratorio2.ipynb, que se debe ejecutar sin errores desde la primera celda a la última. Todo el código debe estar en el mismo notebook, el código debe estar comentado y testeado, el notebook debe estar escrito en forma de informe técnico, escribiendo una celda markdown antes o después de cada celda de código que arroja algún output. 
#%% [markdown]
# # Laboratorio 2: Visualización
# 
# El análisis exploratorio de los datos (EDA) es uno de los primeros pasos en un proyecto de minería de datos luego de la comprensión del problema/negocio y la selección de los datos. Es de suma importancia ya que permite corroborar hipótesis, detectar outliers, estructurar el modelamiento de los datos, resolver problemas de los datos e iterar en el entendimiento del problema.
# 
# Algunos de los principales desafíos son como visualizar datos no estructurados (como texto), visualizar estructuras de datos complejas como redes y visualizar conjunto de datos de alta dimensionalidad.
#%% [markdown]
# ## 1. Teórico, 2 ptos (0.5ptos c/u)
# 
# 1.1 Por qué la visualización de datos es tan relevante en el análisis exploratorio de datos.
# 
# 1.2 En qué contexto debise utilizar un gráfico de barras, cúando debería utilizar un gráfico de barras porcentual por sobre uno basado en frecuencia (de un ejemplo). 
# 
# 1.3 En qué contexto debise utilizar un histograma, cúando debería utilizar un histograma de densidad por sobre uno basado en frecuencia (de un ejemplo). 
# 
# 1.4 Busque una imagen en internet en el que se haya hecho un mal uso de un gráfico y reportela justificando porque fue mal elaborada dicha visualización (además inserte el link a la visualización).
# Ejemplo: https://www.biobiochile.cl/noticias/nacional/chile/2017/11/07/los-errores-del-grafico-que-mostro-pinera-durante-el-debate.shtml
# 
# ## 2. Aplicación, 4 ptos
# 
# ---
# Considere el problema que enfrenta una entidad financiera que tiene altas tasas de fuga voluntaria. Esta institución no tiene claro cuál es el perfil característico que tienen los clientes fugitivos ni cuáles son las razones por las cuales estos se fugan.
# 
# El gerente general le ha pedido **definir el patrón característico de los clientes fugitivos y de los clientes no fugitivos con el objetivo de definir una serie de políticas comerciales** que permitan retener a estos potenciales clientes fugitivos.
# 
# #### Entregables:
# 
# 2.1 Muestre al menos 3 gráficos que muestren diferencias entre el grupo que se fuga y el que no (3 ptos)
# 
# 2.2 Implemente al menos un gráfico para la visualización de datos multidimensional. La idea es que este gráfico resuma los resultados obtenidos del proceso de análisis exploratorio de los datos (i.e, muestra en una visualización las diferencias en los atributos que caracterizan ambas poblaciones) (1 pto)
# 
# ---
# Variable | Descripción
# ------------- | -------------
# 1. ID | Identificador del cliente
# 2. Genero | Genero del cliente
# 3. Renta | Renta en pesos
# 4. Edad | Edad en años
# 5. NIV_Educ | Nivel educacional
# 6. E_Civil | Estado civil
# 7. COD_OfI | Código de la oficina 
# 8. Ciudad | Ciudad de la oficina
# 9. D_Marzo | Deuda de Marzo
# 10. D_Abril | Deuda de Abril 
# 11. D_Mayo | Deuda de Mayo
# 12. D_Junio | Deuda de Junio 
# 13. D_Julio | Deuda de Julio 
# 14. D_Agosto | Deuda de Agosto 
# 15. D_Septiembre | Deuda de Septiembre
# 16. M_Moroso | Meses en Mora
# 17. Monto | Monto preaprobado 
# 18. Seguro | Seguro de gravamen 
# 19. Fuga | Variable objetivo
#%% [markdown]
# ## 1. Teórico
# 
#%% [markdown]
# 1.1. La estadística descriptivo nos entrega la posibilidad de entender cómo se distribuyen los datos, que no podría evaluar si no se realizan visualizaciones de los datos.
# 
# 1.2 majo esta haciendo un cambio
# 
# 1.3
# 
# 1.4
# 
#%% [markdown]
# ## 2. Aplicación

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
df = pd.read_csv('https://raw.githubusercontent.com/dgarridoa/Lab2_Dataset/master/Fuga_Bancos.csv', index_col=0)
n=3
df.head(n)

#%% [markdown]
# 2.1.1 Fugas según género: Muestra la cantidad de personas que se fugan y que no lo hacen, considerando su género. 

#%%
sns.set()
df_count = df.groupby(['GENERO', 'FUGA']).size().reset_index(name='count')
df_count
sns.barplot(x='FUGA', y='count', hue='GENERO', data=df_count)

#%% [markdown]
# De este gráfico se puede concluir que 
#%% [markdown]
# 2.1.2 

