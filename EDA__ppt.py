

#%%
import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression

#%%
os.chdir("C:/Users/sague/OneDrive/Beauchef/Semestre 12/mineria IN/proyecto")

#%%
df1 = pd.read_csv("C:/Users/sague/OneDrive/Beauchef/Semestre 12/mineria IN/proyecto/CD2016_trazabilidad_20190814193348430.csv")
df2 = pd.read_csv("C:/Users/sague/OneDrive/Beauchef/Semestre 12/mineria IN/proyecto/CD2017_trazabilidad_20190814193340711.csv")
df3 = pd.read_csv("C:/Users/sague/OneDrive/Beauchef/Semestre 12/mineria IN/proyecto/CD2018_trazabilidad_20190814193747738.csv")
#%%
df = pd.concat([df1,df2,df3])

#%%
duracion = plt.hist(df['duracionSesion'].astype(float),range=[0,200])
plt.savefig('duracion.png')
#%%
pd.Series(df['provider']).value_counts().plot('bar')
pd.Series(df['portal']).value_counts().plot('bar')
plt.savefig('tipo_cuenta.png')
#%%


#%%
