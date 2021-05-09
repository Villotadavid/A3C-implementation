import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')

df1=pandas.read_csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Training_data_0_A3C1.csv',delimiter=';')
df2=pandas.read_csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Training_data_2_A3C1.csv',delimiter=';')
df3=pandas.read_csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Training_data_7_A3C1.csv',delimiter=';')
add=[df1,df2,df3]
df=pandas.concat(add)
#Time,Hilo,Episodio,Step,Values,log_prob,Rewards,Remaining_Length,Point,Position,Action,Colision,%CPU,%Memoria,Width,Height

################## MEDIAS POR EPISODIO ###############################
total=df.iloc[-1].Episodio
Muestra_Ep=total
media=np.array([None]*Muestra_Ep)
reward=np.array([None]*Muestra_Ep)
logs=np.array([None]*Muestra_Ep)
length=np.array([None]*Muestra_Ep)
x=np.linspace(0,Muestra_Ep,Muestra_Ep)
for i in range (0,Muestra_Ep):  
    media[i]=np.mean(df.Values[(df.Episodio==i)])
    reward[i]=np.mean(df.Rewards[(df.Episodio==i)])
    logs[i]=np.mean(df.log_prob[(df.Episodio==i)])
    length[i]=np.mean(df.Remaining_Length[(df.Episodio==i)])
    
Success=df.Episodio[(df.Rewards==1)]
print (Success)
        
    
plt.plot(x,media,label='Values')
plt.plot(x,reward,label='Reward')
plt.plot(x,logs,label='logs')
plt.plot(x,length,label='R.Length')
plt.legend(loc="upper left")
plt.show()


#%% ################## PLOT DE UN HILO ###############################

step_num=max(df.Step[(df.Episodio==Muestra_Ep)])
Val=df.Values[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
re=df.Rewards[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
lo=df.log_prob[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
rem=df.Remaining_Length[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
plt.plot(np.linspace(0,len(Val),len(Val)),Val,label='Value')
#plt.plot(np.linspace(0,len(Val),len(Val)),lo,label='log probability')
plt.plot(np.linspace(0,len(Val),len(Val)),rem,label='Remining Length')
plt.legend(loc="upper left")
plt.show()

plt.plot(np.linspace(0,len(Val),len(Val)),re,label='Rewards')
plt.legend(loc="upper left")
plt.show()

#%% ################## HISTOGRAMA ###############################

step_num=max(df.Step[(df.Episodio==Muestra_Ep)])
Actions=df.Action[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
plt.hist(Actions,bins=[0,1,2,3,4,5,6],edgecolor='k'.label='Actions')
plt.legend(loc="upper left")
plt.show()

#%%################## GRÁFICO 3D ###############################
Muestra_Ep=970
puntos=df.Position[(df.Episodio==Muestra_Ep)]
puntos=[x[1:-1].split(',') for x in puntos]
puntos=[[float(x) for x in dato] for dato in puntos]  #[float(x) for x in dato]
puntos=np.array(puntos)

map = plt.figure()
ax = Axes3D(map)
ax.autoscale(enable=True, axis='both', tight=True)
ax.scatter(20, 20, 20, cmap='Greens');
ax.plot(puntos[:,0], puntos[:,1], -puntos[:,2]);
plt.show()

################## PLOT DEUN EPISODIO ###############################

'''step_num=max(df.Step[(df.Episodio==Muestra_Ep)])
Val0=df.Values[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
plt.plot(np.linspace(0,len(Val0),len(Val0)),Val0.label='Episode values')
step_num=int(step_num[0:2])
Rew=np.array([None]*step_num)
log=np.array([None]*step_num)

Value=np.mean(df.log_prob[(df.Episodio==Muestra_Ep)])
reward=np.mean(df.Rewards[(df.Episodio==Muestra_Ep)])
log_prob=np.mean(df.log_prob[(df.Episodio==Muestra_Ep)])'''
