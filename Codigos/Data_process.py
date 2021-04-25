import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')

df=pandas.read_csv('C:/Users/usuario/Desktop/Training_data_8.csv',delimiter=';')

#Time,Hilo,Episodio,Step,Values,log_prob,Rewards,Remaining_Length,Point,Position,Action,Colision,%CPU,%Memoria,Width,Height

################## MEDIAS POR EPISODIO ###############################
Muestra_Ep=300
media=np.array([None]*Muestra_Ep)
reward=np.array([None]*Muestra_Ep)
logs=np.array([None]*Muestra_Ep)
length=np.array([None]*Muestra_Ep)
x=np.linspace(0,300,300)
for i in range (0,Muestra_Ep):  
    media[i]=np.mean(df.Values[(df.Episodio==i)])
    reward[i]=np.mean(df.Rewards[(df.Episodio==i)])
    logs[i]=np.mean(df.log_prob[(df.Episodio==i)])
    length[i]=np.mean(df.Remaining_Length[(df.Episodio==i)])
    
plt.plot(x,media,label='Values')
plt.plot(x,reward,label='Reward')
plt.plot(x,logs,label='logs')
plt.plot(x,length,label='R.Length')
plt.legend(loc="upper left")
plt.show()

################## PLOT DEUN EPISODIO ###############################

step_num=max(df.Step[(df.Episodio==1)])
Val0=df.Values[(df.Episodio==1)&(df.Hilo=='w0')]
plt.plot(np.linspace(0,len(Val0),len(Val0)),Val0)
step_num=int(step_num[0:2])
Rew=np.array([None]*step_num)
log=np.array([None]*step_num)

Value=np.mean(df.log_prob[(df.Episodio==Muestra_Ep)])
reward=np.mean(df.Rewards[(df.Episodio==Muestra_Ep)])
log_prob=np.mean(df.log_prob[(df.Episodio==Muestra_Ep)])

#%% ################## PLOT DE UN HILO ###############################
Episodio=320

step_num=max(df.Step[(df.Episodio==Episodio)])
Val=df.Values[(df.Episodio==Episodio)&(df.Hilo=='w0')]
re=df.Rewards[(df.Episodio==Episodio)&(df.Hilo=='w0')]
lo=df.log_prob[(df.Episodio==Episodio)&(df.Hilo=='w0')]
rem=df.Remaining_Length[(df.Episodio==Episodio)&(df.Hilo=='w0')]
plt.plot(np.linspace(0,len(Val),len(Val)),Val,label='Value')
plt.plot(np.linspace(0,len(Val),len(Val)),re,label='Rewards')
plt.plot(np.linspace(0,len(Val),len(Val)),lo,label='log probability')
#plt.plot(np.linspace(0,len(Val),len(Val)),rem,label='Remining Length')
plt.legend(loc="upper left")
plt.show()
#%%################## GRÁFICO 3D ###############################
puntos=df.Position[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
puntos=[x[1:-1].split(',') for x in puntos]
puntos=[print (dato) for dato in puntos]  #[float(x) for x in dato]
puntos=np.array(puntos)

map = plt.figure()
ax = Axes3D(map)
ax.autoscale(enable=True, axis='both', tight=True)
ax.scatter(50, 50, +40, cmap='Greens');

ax.plot(puntos[:,0], puntos[:,1], -puntos[:,2]);
plt.show()

