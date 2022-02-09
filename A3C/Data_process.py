import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')

#df1=pandas.read_csv('C:/Users/usuario/Desktop/Doctorado/Analisis/2_threads/Training_data_0-340.csv',delimiter=';')
#df2=pandas.read_csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Linearvel=06/Training_data_6_6_Threads.csv',delimiter=';')
#add=[df1,df2]
#df=pandas.concat(add)'''
#df=pandas.read_csv('C:/Users/usuario/Desktop/Doctorado/Analisis/6_threads/Training_data_2.csv',delimiter=';')

df=pandas.read_csv('C:/Users/usuario/Desktop/Doctorado/A3C_Implementation/A3C/Training_data_0.csv',delimiter=';')

#Time,Hilo,Episodio,Step,Values,log_prob,Rewards,Remaining_Length,Point,Position,Action,Colision,%CPU,%Memoria,Width,Height

################## MEDIAS POR EPISODIO ###############################

total=int(df.iloc[-1].Episodio)
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
Fail=df.Episodio[(df.Rewards==-1)]
print ('Success')
print (Success)
print ('Fail')
print (Fail)
                
    
plt.plot(x,length,label='R.Length')
plt.legend(loc="upper left")
plt.show()

plt.plot(x,media,label='Values')
plt.legend(loc="upper left")
plt.show()

plt.plot(x,reward,label='Reward')
plt.legend(loc="upper left")
plt.show()


#%% ################## PLOT DE UN HILO ###############################
'''
Muestra_Ep=2822
#2400/w5
#988/w1
Hilo='w0'

Val=df.Values[(df.Episodio==Muestra_Ep)&(df.Hilo==Hilo)]
re=df.Rewards[(df.Episodio==Muestra_Ep)&(df.Hilo==Hilo)]
lo=df.log_prob[(df.Episodio==Muestra_Ep)&(df.Hilo==Hilo)]
rem=df.Remaining_Length[(df.Episodio==Muestra_Ep)&(df.Hilo==Hilo)]



plt.subplot(2,1,1)
plt.plot(np.linspace(0,len(Val),len(Val)),Val,label='Value')
plt.plot(np.linspace(0,len(Val),len(Val)),rem,label='Remaining Length (m)')
plt.legend(loc="upper left")


plt.subplot(2,1,2)
plt.xlabel('Step',size=14)
plt.plot(np.linspace(0,len(Val),len(Val)),re,label='Rewards',color='orange')
plt.legend(loc="lower left")
plt.show()
'''
#%% ################## HISTOGRAMA ###############################
'''

step_num=max(df.Step[(df.Episodio==Muestra_Ep)])
Actions=df.Action[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
plt.hist(Actions,bins=[0,1,2,3,4,5,6],edgecolor='k',label='Actions')
plt.legend(loc="upper left")
plt.show()
'''
#%%################## GRÁFICO 3D ###############################

Muestra_Ep=2822

puntos0=df2.Position[(df2.Episodio==Muestra_Ep) & (df2.Hilo=='w0')]
puntos1=df2.Position[(df2.Episodio==Muestra_Ep) & (df2.Hilo=='w1')]
puntos2=df2.Position[(df2.Episodio==Muestra_Ep) & (df2.Hilo=='w2')]
puntos3=df2.Position[(df2.Episodio==Muestra_Ep) & (df2.Hilo=='w3')]
puntos4=df2.Position[(df2.Episodio==Muestra_Ep) & (df2.Hilo=='w4')]
puntos5=df2.Position[(df2.Episodio==Muestra_Ep) & (df2.Hilo=='w5')]

puntos0=[str(dato).replace('   ',' ') for dato in puntos0]
puntos0=[str(dato).replace('  ',' ') for dato in puntos0]
puntos0=[str(dato).replace('[  ','') for dato in puntos0]
puntos0=[str(dato).replace('[ ','') for dato in puntos0]
puntos0=[str(dato).replace(' ]','') for dato in puntos0]
puntos0=[str(dato).replace(']','') for dato in puntos0]
puntos0=[str(dato).replace('  ',' ') for dato in puntos0]
puntos0=[x.split(' ') for x in puntos0]

puntos0=[[float(x) for x in dato] for dato in puntos0]  #[float(x) for x in dato]
puntos0=np.array(puntos0)

puntos1=[str(dato).replace('   ',' ') for dato in puntos1]
puntos1=[str(dato).replace('  ',' ') for dato in puntos1]
puntos1=[str(dato).replace('[  ','') for dato in puntos1]
puntos1=[str(dato).replace('[ ','') for dato in puntos1]
puntos1=[str(dato).replace(' ]','') for dato in puntos1]
puntos1=[str(dato).replace(']','') for dato in puntos1]
puntos1=[str(dato).replace('  ',' ') for dato in puntos1]
puntos1=[x.split(' ') for x in puntos1]

puntos1=[[float(x) for x in dato] for dato in puntos1]  #[float(x) for x in dato]
puntos1=np.array(puntos1)

puntos2=[str(dato).replace('   ',' ') for dato in puntos2]
puntos2=[str(dato).replace('  ',' ') for dato in puntos2]
puntos2=[str(dato).replace('[  ','') for dato in puntos2]
puntos2=[str(dato).replace('[ ','') for dato in puntos2]
puntos2=[str(dato).replace(' ]','') for dato in puntos2]
puntos2=[str(dato).replace(']','') for dato in puntos2]
puntos2=[str(dato).replace('  ',' ') for dato in puntos2]
puntos2=[x.split(' ') for x in puntos2]

puntos2=[[float(x) for x in dato] for dato in puntos2]  #[float(x) for x in dato]
puntos2=np.array(puntos2)

puntos3=[str(dato).replace('   ',' ') for dato in puntos3]
puntos3=[str(dato).replace('  ',' ') for dato in puntos3]
puntos3=[str(dato).replace('[  ','') for dato in puntos3]
puntos3=[str(dato).replace('[ ','') for dato in puntos3]
puntos3=[str(dato).replace(' ]','') for dato in puntos3]
puntos3=[str(dato).replace(']','') for dato in puntos3]
puntos3=[str(dato).replace('  ',' ') for dato in puntos3]
puntos3=[x.split(' ') for x in puntos3]

puntos3=[[float(x) for x in dato] for dato in puntos3]  #[float(x) for x in dato]
puntos3=np.array(puntos3)

puntos4=[str(dato).replace('   ',' ') for dato in puntos4]
puntos4=[str(dato).replace('  ',' ') for dato in puntos4]
puntos4=[str(dato).replace('[  ','') for dato in puntos4]
puntos4=[str(dato).replace('[ ','') for dato in puntos4]
puntos4=[str(dato).replace(' ]','') for dato in puntos4]
puntos4=[str(dato).replace(']','') for dato in puntos4]
puntos4=[str(dato).replace('  ',' ') for dato in puntos4]
puntos4=[x.split(' ') for x in puntos4]

puntos4=[[float(x) for x in dato] for dato in puntos4]  #[float(x) for x in dato]
puntos4=np.array(puntos4)

puntos5=[str(dato).replace('   ',' ') for dato in puntos5]
puntos5=[str(dato).replace('  ',' ') for dato in puntos5]
puntos5=[str(dato).replace('[  ','') for dato in puntos5]
puntos5=[str(dato).replace('[ ','') for dato in puntos5]
puntos5=[str(dato).replace(' ]','') for dato in puntos5]
puntos5=[str(dato).replace(']','') for dato in puntos5]
puntos5=[str(dato).replace('  ',' ') for dato in puntos5]
puntos5=[x.split(' ') for x in puntos5]

puntos5=[[float(x) for x in dato] for dato in puntos5]  #[float(x) for x in dato]
puntos5=np.array(puntos5)

map = plt.figure()
ax = Axes3D(map)
ax.autoscale(enable=True, axis='both', tight=True)
ax.scatter(20, 20, 20, cmap='Greens');
ax.plot(puntos0[:,0], puntos0[:,1], -puntos0[:,2],color='blue',label='Thread 1');
ax.plot(puntos1[:,0], puntos1[:,1], -puntos1[:,2],color='orange',label='Thread 2');
ax.plot(puntos2[:,0], puntos2[:,1], -puntos2[:,2],color='green',label='Thread 3');
ax.plot(puntos3[:,0], puntos3[:,1], -puntos3[:,2],color='purple',label='Thread 4');
ax.plot(puntos4[:,0], puntos4[:,1], -puntos4[:,2],color='yellow',label='Thread 5');
ax.plot(puntos5[:,0], puntos5[:,1], -puntos5[:,2],color='red',label='Thread 6');
#ax.plot(puntos5[:,0], puntos1[:,1], -puntos1[:,2],color='purple');
plt.legend(loc="upper left")
plt.show()


################## PLOT DEUN EPISODIO ###############################
'''
Muestra_Ep=2822
step_num=max(df.Step[(df.Episodio==Muestra_Ep)])
Val0=df.Values[(df.Episodio==Muestra_Ep)&(df.Hilo=='w0')]
plt.plot(np.linspace(0,len(Val0),len(Val0)))
step_num=int(step_num[0:2])
Rew=np.array([None]*step_num)
log=np.array([None]*step_num)

Value=np.mean(df.log_prob[(df.Episodio==Muestra_Ep)])
reward=np.mean(df.Rewards[(df.Episodio==Muestra_Ep)])
log_prob=np.mean(df.log_prob[(df.Episodio==Muestra_Ep)])


map = plt.figure()
ax = Axes3D(map)
ax.autoscale(enable=True, axis='both', tight=True)

plt.show()
'''