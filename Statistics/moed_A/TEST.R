install.packages("readxl")
library(readxl)

final1data1  = read_excel("C:/Users/bechina/Desktop/final1data1.xlsx", sheet = "Sheet1")


###2
t.test(final1data1$A,final1data1$B,alternative = "two.sided",var.equal = T)
###4

final1data2  = read_excel("C:/Users/bechina/Desktop/final1data2.xlsx", sheet = "Sheet2")
#analyze the km (the mean that we check) by company
faov=aov(final1data2$val~final1data2$type)
summary(faov) #p.value = 0.576 - Not Reject
TukeyHSD(faov)
install.packages("DescTools")
library(DescTools)
ScheffeTest(faov)
# order the companies by abc: (m.company A + m.company D)/2 = (m.company B + m.company C)/2
ScheffeTest(faov, contrasts=c(-1,0,1/2,-1/2))

###5
head(mtcars)
data=(mtcars)[,c(2,3,4,5,6,7,8,9,10,11)] #without goal column
head(data)
cordata = cor(data)
cordata
e=eigen(cordata)
summary(e)
e$vectors #if we want to create new row to data multiple 0.52*feature1, -0.26*feature2, 0.58*feature 3, 0.56*feature 4
e$values
#the percent that the 2 first features of the PCA "hold"
(e$values[1]+e$values[2])/sum(e$values)
cccc=prcomp(data,scale=T)
cccc$sdev^2  ### Eigenvalues
cccc$rotation ### Eigenvectors
cccc$x ### the values after the PCA
cor(cccc$x)  ### there is no correlation - like 0, we can use them to predict.
c=cccc$sdev^2

for (i in 1:10){print(sum(c[1:i])/sum(c))}



###6
# 11.5 ~~~~~~~~~~~~~~~~~~
# LDA - with 3 Classes
install.packages("ISLR")
library(ISLR)
data=iris
head(data)
install.packages("MASS")
library(MASS)

countrows = as.double(nrow(data))
table(data)

lda.model=lda(Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,data=data)
lda.model
#predictions
lda.pred=predict(lda.model,data[,1:4])
lda.pred$posterior
lda.pred$class
table(lda.pred$class,data[,5])
mean(lda.pred$class==data[,5])

qda.model=qda(Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,data=data)
qda.pred=predict(qda.model,data[,1:4])
qda.pred$posterior
qda.pred$class
table(qda.pred$class,data[,5])
mean(qda.pred$class==data[,5])

###8
wilcox.test(final1data1$A,final1data1$B,alternative = "two.sided",conf.level = TRUE)
t.test(final1data1$A,final1data1$B,alternative = "two.sided",var.equal = T)
###3
q2mat <- matrix( ncol =3,nrow = 2,c(150,100,100,400,50,200))
colnames(q2mat)<- c("Helth","Reg","Junk")
rownames(q2mat) <- c("Leader","Not")
temp=chisq.test(q2mat)
temp$expected
temp$p.value


###7
# 9.2 ~~~~~~~~~~~~~~~~~~
#Stepwise regression (aic) - forward
data=mtcars

intercept_only=lm(mpg~1,data=data)
summary(intercept_only)
model_cyl=lm(mpg~.-cyl,data=data)
# summary(model2)
model2=lm(mpg~.,data=data)
forward=step(intercept_only,direction = "forward",scope=formula(model2),trace = 1)
backwards=step(model2,direction = "backward",scope=formula(model2),trace = 1)
both=step(intercept_only,direction = "both",scope=formula(model2),trace = 1)


summary(model_cyl)

summary(forward)
anova(forward)

model2=lm(mpg~.,data=data)
summary(model2)
model_test=lm(mpg~.-cyl-disp,data=data)
summary(model_test)



ssr_full = sum((fitted(model2) - mean(data$mpg))^2)
ssr_paritally = sum((fitted(model_test) - mean(data$mpg))^2)
mse_full =  sum(residuals(model2)^2)/(nrow(data)-ncol(data))
derduce_features_num = 2
F_statistic = ((ssr_full-ssr_paritally)/derduce_features_num)/mse_full
F_critical = qf(0.95, df1=derduce_features_num, df2 = (nrow(data)-ncol(data))) 
F_statistic
F_critical
print ("If the top num is bigger than the second, Reject H0 - The Features have a predictive value above the predictive value of the rest of the independent variables.")
