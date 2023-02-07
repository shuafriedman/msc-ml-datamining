### T-Tests###

#problem 1: the following is data  of 200 emplyees representing the amount of
#           time needed to complete a  certain task. I think the mean time is 11 minutes but you think I am wrong. 
#           Test alpha=0.05
t.test(homework2$time,mu=11,alternative = "two.sided")
sd(homework2$time)
#problem 2: 2-sample t test with alpha=0.05. (2-sided) (unpaired and paired)
t.test(t_test_data7$before, t_test_data7$after,alternative = "greater", var.equal = TRUE)
# problem 3
t.test(t_test_data7$before, t_test_data7$after,alternative = "greater", var.equal = TRUE, paired=TRUE)
#


####one sample test of variance-- Chisquared ####

# Estimate the variance, test what the null hypothesis is using the chi-squared 
# test that the variance is  equal to a user-specified value,
# and create a confidence interval for the variance.
install.packages("EnvStats")
library(EnvStats)
#sigma.squared-- a numeric scalar indicating the hypothesized value of the variance. 
#              The default value is sigma.squared=1.
varTest(chidata1$apples,alternative = "greater", sigma.squared=3)


#### 2 sample test of Variance-- F-test###:  

## F test to compare the variances of two samples from normal populations. 
# The null hypothesis is that the  ratio of the variances of the populations from which 
# x and y were drawn, or in the data to which the linear models x and y were fitted,
# is equal to ratio (default = 1)
str(fdata1)
var(fdata1$tal,na.rm =TRUE)
var(fdata1$lev,na.rm =TRUE)
var.test(fdata1$tal,fdata1$lev,alternative = "greater" , conf.level = 0.95)


#### goodness of fit: (X is a vector)####

chidata2
chisq.test(chidata2$o1,p=c(1/6,1/6,1/6,1/6,1/6,1/6))
qchisq(.95,5)
chisq.test(chidata2$o2,p=c(1/6,1/6,1/6,1/6,1/6,1/6))

####kollmogorov smirnov test (better for testing fit to continous distribution (CDF)### 

# a two-sample (Smirnov) test of the null hypothesis that x and y were drawn from 
# the same continuous distribution.
# y arguement --- either a numeric vector of data values, or a character string naming a cumulative 
#                 distribution function or an actual cumulative distribution function such as pnorm. 
#                  Only continuous CDFs are valid.
ksdata1
ksdata1$oranges=(ksdata1$Apples-100)/20
ks.test(ksdata1$oranges,"pnorm")
ks.test(ksdata1$oranges, ksdata1$Apples) #two sample Ks test (are both samples from same distribution)

####contingency table anaysis (x is a matrix)####

# Use contingency tables to understand the relationship between categorical variables.
# For example, is there a relationship between gender (male/female) 
# and type of computer (Mac/PC)?
datact=as.matrix(chidata3)
colnames(datact)<- c("rep","dem","ind")
rownames(datact)=c("cau","afro","asian")
datact
chisq.test(datact)

###one way anova###

# The one-way analysis of variance (ANOVA) is used to determine whether there are 
# any statistically significant differences between the means of three or more 
# independent (unrelated) groups.
#### ***** The Student's t test is used to compare the means between two groups, 
####### whereas ANOVA is used to compare the means among three or more groups.
str(fdata2)
faov=aov(fdata2$km~fdata2$company)
summary(faov)
TukeyHSD(faov)
install.packages("DescTools")
library(DescTools)
ScheffeTest(faov)
ScheffeTest(faov, contrasts=c(1,-1/3,-1/3,-1/3))

#2 way anova: Two-way ANOVA test is used to evaluate simultaneously the effect of 
#             two grouping variables (A and B) on a response variable.
#             Not the above fitted model is called additive model. It makes an assumption 
              #that the two factor variables are independent. If you think that these two 
#             variables might interact to create an synergistic effect, replace 
#             the plus symbol (+) by an asterisk (*), as follow.
faov2=aov(fdata2$km~fdata2$company+fdata2$place+fdata2$company*fdata2$place)
summary(faov2)

