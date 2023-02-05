##Chi-Squared Test on Variance
## A new methd of feeding was tested on 12 animals, and gave the following weights in grams:
weights = c(1521, 1367, 1251, 1215, 1064, 1138, 1292, 1140, 988, 912, 836, 1216)
## Test whether the variance of the weights is:
sigmaSquared = 40000

EnvStats::varTest(weights, sigma.squared=sigmaSquared, alternative='two.sided' )

## F-test for comparing variances
## Check whether there is a difference between Mens and Womens hourly wages:
women = c(34,69,32,43,45)
men  = c(70, 85, 78, 56)
var.test(men, women, alternative = 'two.sided')

## Goodness of Fit
observed = c(94, 93, 112, 101, 104, 95, 100, 99, 108, 94)
prob = rep(1/length(observed), length(observed))
chisq.test(observed, p=prob)

##Test of Independence: Contingency Tables
#Determine whether gender and physical fitness are dependent or independent variables:
low = c(10, 15)
average = c(15, 20)
high = c(10,20)
data = data.frame(low, average, high, row.names = c('Male', 'Female') )
data
chisq.test(data)