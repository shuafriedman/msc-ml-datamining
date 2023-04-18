### LDA ###
# LinearDiscriminantAnalysis can be used to perform supervised dimensionality reduction, 
# by projecting the input data to a linear subspace consisting of the directions which 
# maximize the separation between classes (in a precise sense discussed in the 
# mathematics section below). 

# The dimension of the output is necessarily less than the number of classes, so this is 
# in general a rather strong dimensionality reduction, and only makes sense in a 
# multiclass setting.

install.packages("ISLR")
library(ISLR)
data=Smarket
library(MASS)
#split the data to before/after 2005
train=(data$Year<2005)
table(train)
datatrain=data[train,]
datatest=data[!train,]
colnames(datatrain)
lda.model=lda(Direction~Lag1+Lag2,data=datatrain)
lda.model
#predictions
lda.pred=predict(lda.model,datatest[,1:8])
lda.pred$class
table(lda.pred$class,datatest[,9])
mean(lda.pred$class==datatest[,9])
#qda-- Quadratic Dicrimminant Analysis
qda.model=qda(Direction~Lag1+Lag2,data=datatrain)
qda.model
qda.pred=predict(qda.model,datatest[,1:8])
qda.pred$class
table(qda.pred$class,datatest[,9])
mean(qda.pred$class==datatest[,9])
