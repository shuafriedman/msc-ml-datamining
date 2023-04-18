#dataset=hitters, program = pls
# install.packages('ISLR')
# library(ISLR)
# head(Hitters)
# str(Hitters)
# 
# data1=na.omit(Hitters) ### Can also set na.action in the prcomp formula below
# a=c(1:13,16,17,18)
# b=prcomp(data1[,a],scale=TRUE) ### PRincipal Component Analysis
# c=b$sdev^2 #get the variances
# c
# for (i in 1:16){print(sum(c[1:i])/sum(c))} #cumulative sum of the variances for each feature

data(mtcars)
my_pca <- prcomp(mtcars, scale=TRUE, center= TRUE, retx=T)
summary(my_pca)
my_pca$sdev

# Compute variance
my_pca.var <- my_pca$sdev ^ 2
propve <- my_pca.var / sum(my_pca.var)
propve
which(cumsum(propve) >= 0.9)[1]
train.data <- data.frame(disp = mtcars$disp, my_pca$x[, 1:which(cumsum(propve) >= 0.9)[1]])
train.data



# fitting the model-- Partial Least Squares regression with PCA (built into pcr function)
install.packages('pls')
library(pls)
set.seed(2)
pcr.fit=pcr(Salary~.,data=data1,scale=TRUE)
summary(pcr.fit)
validationplot(pcr.fit,val.type = "MSEP")
#train test
set.seed(1)
train=sample(1:263,263/2)
test=(-train)
data2.train=data1[train,]
data2.test=data1[test,]
pcr.fit=pcr(Salary~.,data=data1,subset=train,scale=TRUE)
#predictions
pcr.predict=predict(pcr.fit,data2.test,ncomp=5)
mean((pcr.predict-data2.test$Salary)^2)
#Compare to lm regression
lm.fit=lm(Salary~.,data=data2.train)
lm.predict=predict(lm.fit,data2.test)
mean((lm.predict-data2.test$Salary)^2)
