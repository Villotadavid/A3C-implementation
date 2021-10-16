library(ggplot2)
library(dplyr)
df3=read.csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Linearvel=06/Training_data_0_3_Threads.csv',sep=';')
df1=read.csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Linearvel=06/Training_data_4_1_Threads.csv',sep=';')
df2=read.csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Linearvel=06/Training_data_3_2_Threads.csv',sep=';')
df5=read.csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Linearvel=06/Training_data_0_6_g.csv',sep=';')
df6=read.csv('C:/Users/usuario/Desktop/Doctorado/Analisis/Linearvel=06/Training_data_5_6_Threads_n.csv',sep=';')

##############VALUES##########################################

Values3<-df3 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Values))

Values1<-df1 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Values))

Values6<-df6 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Values))

Values5<-df5 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Values))

Values2<-df2 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Values))



ggplot(data=Values3,aes(x=Episodio,y=datos))+geom_jitter(data=Values3,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.05,color='red')+
  geom_jitter(data=Values1,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.05,color='green')+
  geom_jitter(data=Values6,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.05,color='purple')+
  geom_jitter(data=Values2,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.05,color='blue')+
  geom_jitter(data=Values5,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.05,color='blue')+
  geom_smooth(data=Values1,aes(x=Episodio,y=datos),color='green')+geom_smooth(data=Values3,aes(x=Episodio,y=datos),color='red')+
  geom_smooth(data=Values6,aes(x=Episodio,y=datos),color='purple')+ geom_smooth(data=Values2,aes(x=Episodio,y=datos),color='blue')+
  geom_smooth(data=Values5,aes(x=Episodio,y=datos),color='orange')+
  theme(legend.position = "bottom")


ggplot(data=Rewards3,aes(x=Episodio,y=datos))+geom_jitter(height = 2, width = 2,alpha=0.1,color='red')+
  geom_smooth()

ggplot(data=rlength3,aes(x=Episodio,y=datos))+geom_jitter(height = 2, width = 2,alpha=0.1,color='red')+
  geom_smooth()


##############REWARDS#########################################

Rewards3<-df3 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Rewards))

Rewards1<-df1 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Rewards))

Rewards6<-df6 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Rewards))

Rewards2<-df2 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Rewards))



ggplot()+geom_jitter(data=Rewards3,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color='red')+
  geom_jitter(data=Rewards1,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color='green')+
  geom_jitter(data=Rewards6,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color='purple')+
  geom_jitter(data=Rewards2,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color='blue')+
  geom_smooth(data=Rewards1,aes(x=Episodio,y=datos),color='green')+geom_smooth(data=Rewards3,aes(x=Episodio,y=datos),color='red')+
  geom_smooth(data=Rewards6,aes(x=Episodio,y=datos),color='purple')+ geom_smooth(data=Rewards2,aes(x=Episodio,y=datos),color='blue')+
  theme(legend.position = "bottom")

Rewardsdf<-c(Rewards2[c(1)],datos1=Rewards1[c(2)],datos2=Rewards2[c(2)],datos3=Rewards3[c(2)],datos6=Rewards6[c(2)])
ggplot()+geom_jitter(data=Rewardsdf,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color=y)
###########REMAINING LENGTH###################################

Remaining_Length3<-df3 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Remaining_Length))

Remaining_Length1<-df1 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Remaining_Length))

Remaining_Length6<-df6 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Remaining_Length))

Remaining_Length2<-df2 %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Remaining_Length))

Remaining_Length1<-Remaining_Length1-10
Remaining_Length2<-Remaining_Length2-10
Remaining_Length3<-Remaining_Length3-10
Remaining_Length6<-Remaining_Length6-10

ggplot()+geom_jitter(data=Remaining_Length3,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color='red')+
  geom_jitter(data=Remaining_Length1,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color='green')+
  geom_jitter(data=Remaining_Length6,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color='purple')+
  geom_jitter(data=Remaining_Length2,aes(x=Episodio,y=datos), height = 2, width = 2,alpha=0.1,color='blue')+
  geom_smooth(data=Remaining_Length1,aes(x=Episodio,y=datos),color='green')+geom_smooth(data=Remaining_Length3,aes(x=Episodio,y=datos),color='red')+
  geom_smooth(data=Remaining_Length6,aes(x=Episodio,y=datos),color='purple')+ geom_smooth(data=Remaining_Length2,aes(x=Episodio,y=datos),color='blue')+
  theme(legend.position = "bottom")

###########REMAINING LENGTH###################################
Actions1<-df3$Episodio[df3$Action]
tipo<-factor(rep(c("Acciones"), each=Actions1))
data<-data.frame(Tipo=tipo,DATOS=Actions1)
ggplot(data=data,aes(x=Actions1,fill=Tipo))+geom_histogram(alpha=0.35, position="identity")

Actions2<-df3$Episodio[df2$Actions]
Actions3<-df3$Episodio[df3$Actions]
Actions6<-df3$Episodio[df6$Actions]

##########SUCCESS#############################################

Success1<-df3$Episodio[df1$Rewards==1]
Success2<-df3$Episodio[df2$Rewards==1]
Success3<-df3$Episodio[df3$Rewards==1]
Success6<-df3$Episodio[df6$Rewards==1]

long<-length(Success6)
tipo<-factor(rep(c("Eje_X"), each=long))
Success<-c(Success6)
data<-data.frame(Tipo=tipo,DATOS=Success6)

ggplot(data=data,aes(x=Success6,fill=Tipo))+geom_histogram(alpha=0.35, position="identity")