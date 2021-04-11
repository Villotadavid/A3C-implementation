import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')

df=pandas.read_csv('C:/Users/usuario/Desktop/Training_data_3.csv',delimiter=';')

#Time,Hilo,Episodio,Step,Values,log_prob,Rewards,Remaining_Length,Point,Position,Action,Colision,%CPU,%Memoria,Width,Height

################## MEDIAS POR EPISODIO ###############################
Muestra_Ep=700
media=np.array([None]*Muestra_Ep)
reward=np.array([None]*Muestra_Ep)
logs=np.array([None]*Muestra_Ep)
length=np.array([None]*Muestra_Ep)
x=np.linspace(0,700,700)
for i in range (0,Muestra_Ep):  
    media[i]=np.mean(df.Values[(df.Episodio==i)])
    reward[i]=np.mean(df.Rewards[(df.Episodio==i)])
    logs[i]=np.mean(df.log_prob[(df.Episodio==i)])
    length[i]=np.mean(df.Remaining_Length[(df.Episodio==i)])
    
'''plt.plot(x,media)
plt.plot(x,reward)
plt.plot(x,logs)
plt.plot(x,length)'''

################## PLOT DEUN EPISODIO ###############################

step_num=max(df.Step[(df.Episodio==1)])
Val0=df.Values[(df.Episodio==1)&(df.Hilo=='w0')]
Val1=df.Values[(df.Episodio==1)&(df.Hilo=='w1')]
Val2=df.Values[(df.Episodio==1)&(df.Hilo=='w2')]
Val3=df.Values[(df.Episodio==1)&(df.Hilo=='w3')]
plt.plot(np.linspace(0,len(Val0),len(Val0)),Val0)
plt.plot(np.linspace(0,len(Val1),len(Val1)),Val1)
plt.plot(np.linspace(0,len(Val2),len(Val2)),Val2)
plt.plot(np.linspace(0,len(Val3),len(Val3)),Val3)
Rew=np.array([None]*step_num)
log=np.array([None]*step_num)

Value=np.mean(df.log_prob[(df.Episodio==700)])
reward=np.mean(df.Rewards[(df.Episodio==700)])
log_prob=np.mean(df.log_prob[(df.Episodio==700)])

#%% ################## PLOT DE UN HILO ###############################
Episodio=1

step_num=max(df.Step[(df.Episodio==Episodio)])
Val=df.Values[(df.Episodio==Episodio)&(df.Hilo=='w0')]
re=df.Rewards[(df.Episodio==Episodio)&(df.Hilo=='w0')]
lo=df.log_prob[(df.Episodio==Episodio)&(df.Hilo=='w0')]
rem=df.Remaining_Length[(df.Episodio==Episodio)&(df.Hilo=='w0')]
plt.plot(np.linspace(0,len(Val),len(Val)),Val,label='Value')
plt.plot(np.linspace(0,len(Val),len(Val)),re,label='Rewards')
plt.plot(np.linspace(0,len(Val),len(Val)),lo,label='log probability')
plt.plot(np.linspace(0,len(Val),len(Val)),rem,label='Remining Length')
plt.legend(loc="upper left")

#%%################## GRÁFICO 3D ###############################
puntos=df.Position[(df.Episodio==700)&(df.Hilo=='w1')]
puntos=[x[1:-1].split(',') for x in puntos]
puntos=[[float(x) for x in dato] for dato in puntos]
puntos=np.array(puntos)

map = plt.figure()
ax = Axes3D(map)
ax.autoscale(enable=True, axis='both', tight=True)
ax.scatter(50, 50, +40, cmap='Greens');
ax.plot(puntos[:,0], puntos[:,1], -puntos[:,2]);
plt.show()

