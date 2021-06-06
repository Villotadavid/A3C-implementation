library(ggplot2)
library(dplyr)
dfx=read.csv('C:/Users/usuario/Desktop/Doctorado/Analisis/2_threads/Training_data_0.csv',sep=';')

##############################################################

Values<-dfx %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Values))

Rewards<-dfx %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Rewards))

rlength<-dfx %>%
  group_by(Episodio) %>%
  summarise(datos = mean(Remaining_Length))

ggplot(data=Values,aes(x=Episodio,y=datos))+geom_jitter(height = 2, width = 2,alpha=0.1,color='red')+
  geom_smooth()

ggplot(data=Rewards,aes(x=Episodio,y=datos))+geom_jitter(height = 2, width = 2,alpha=0.1,color='red')+
  geom_smooth()

ggplot(data=rlength,aes(x=Episodio,y=datos))+geom_jitter(height = 2, width = 2,alpha=0.1,color='red')+
  geom_smooth()

##############################################################

Success<-dfx$Episodio[dfx$Rewards==1]
long<-length(Success)
tipo<-factor(rep(c("Eje_X"), each=long))
Success<-c(Success)
data<-data.frame(Tipo=tipo,DATOS=Success)

ggplot(data=data,aes(x=datos,fill=Tipo))+geom_histogram(alpha=0.35, position="identity")