#one sample test of variance

my.data=hw3_1
install.packages("EnvStats")
library(EnvStats)
varTest(my.data$apples,alternative = "two.sided", sigma.squared=225)
#2 sample test
str(my.data)
var(my.data$apples,na.rm =TRUE)
var(my.data$oranges,na.rm =TRUE)
var.test(my.data$apples,my.data$oranges,alternative = "two.sided" , conf.level = 0.95)
#goodness of fit
chisq.test(solutions3_4$counts,p=c(.032,.302,.507,.159))
qchisq(.95,5)
#contingincy table anaysis
solutions3_4b
my_data2=solutions3_4b[1:4,1:4]
datact=as.matrix(my_data2)
dim(datact)
colnames(datact)<- c("har","dati","mes","hilo")
rownames(datact)=c("north","mercaz","jer","south")
datact
temp=chisq.test(datact)
temp$expected


#one way anova
data = read_ods("/home/shua/Desktop/msc-ml-datamining/Statistics/Anova/data.ods")
stack = stack(data)
stack

# my.data3=solutions3_4_2
# my.data3
# str(my.data3)
# faov=aov(my.data3$value~my.data3$company)
# summary(faov)

faov = aov(values~ind, data=stack)
faov
summary(faov)
TukeyHSD(faov)

install.packages("DescTools")
library(DescTools)
ScheffeTest(faov)
ScheffeTest(faov, contrasts=c(.25,.25,.25,.25,-1))
#tal

#2 way anova
my.data4=solutions3_4d
faov2=aov(my.data4$response~my.data4$gender)
summary(faov2)
faov2=aov(my.data4$response~my.data4$drug)
summary(faov2)
faov2=aov(my.data4$response~my.data4$gender+my.data4$drug+my.data4$gender*my.data4$drug)
summary(faov2)