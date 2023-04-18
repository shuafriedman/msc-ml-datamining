#logistic regression
install.packages('mlbench')
library(mlbench)
data(BreastCancer)
head(BreastCancer)
bc=BreastCancer
anyNA(bc)
bc=na.omit(bc)
table(bc$Class)
str(bc)
#remove id col
bc=bc[,-1]
#converting data to numeric
for (i in 1:9){
  bc[,i]=as.numeric(bc[,i])
}
str(bc)
bc$Class=ifelse(bc$Class=='malignant',1,0) #convert the target column to numeric (binary)
head(bc$Class)
bc$Class=as.factor(bc$Class)
str(bc)
#train test split
train=sample(1:683,.7*683)
y.train=bc[train,10]
table(y.train) 
y.test=bc[-train,10]    
bctrain=bc[train,]
bctest=bc[-train,]
#logistic model
logitmodel=glm(Class~Cl.thickness+Cell.size+Cell.shape,data=bctrain,family = 'binomial')
summary(logitmodel)
#testing the model
pred=predict(logitmodel,bctest[,1:9],type='response')
head(pred) #returns the probabilities
y.pred=as.factor(ifelse(pred>.5,1,0)) #labels based on athreshold
head(y.pred)
mean(y.pred==y.test)
table(y.pred,y.test)
