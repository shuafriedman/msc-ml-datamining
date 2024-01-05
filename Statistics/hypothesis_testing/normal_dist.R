#### Pnorm ####

# Question: Suppose widgit weights produced at Acme Widgit Works have weights that are 
# normally distributed with mean 17.46 grams and variance 375.67 grams.
# What is the probability that a randomly chosen widgit weighs more then 19 grams?
#   
#   Question Rephrased: What is P(X > 19) when X has the N(17.46, 375.67) distribution?
#   
#   Caution: R wants the s. d. as the parameter, not the variance. 
#            We'll need to take a square root!
# 
1 - pnorm(19, mean=17.46, sd=sqrt(375.67))

#### Qnorm ###

# is the R function that calculates the inverse c. d. f. F-1 of the normal 
# distribution The c. d. f. and the inverse c. d. f. are related by
# p = F(x)
# x = F-1(p)
# So given a number p between zero and one, qnorm looks up the p-th quantile of the 
#normal distribution. As with pnorm, optional arguments specify the mean and standard 
#deviation of the distribution.

# Example
# Question: Suppose IQ scores are normally distributed with mean 100 and standard 
# deviation 15. What is the 95th percentile of the distribution of IQ scores?
#   
#### Question Rephrased: What is F-1(0.95) when X has the N(100, 152) distribution?
qnorm(0.95, mean=100, sd=15)

### HW Example with chisq###
# . A thousand people are asked there ages. The following are the results, alpha=.05				
# less then 20	   20-29.9	30-40	  more than 40
#          200   	  350      350	      100
# 
# does this data come from a population where ages are 
# normally distributed with mean = 33 and standard deviation =7
tar3_q1_data
p_less_then_20 <- pnorm(20,33,7)
p_less_then_20
p_20_to_30 <- pnorm(30,33,7) - pnorm(20,33,7)
p_20_to_30
p_30_to_40<- pnorm(40,33,7) - pnorm(30,33,7)
p_30_to_40
p_more_then_40 <- 1 - pnorm(40,33,7)
p_more_then_40
#chisq.test(tar3_q1_data$num,p=c(0.032,0.302,0.507,0.159))
chisq.test(tar3_q1_data$num,p=c(p_less_then_20,p_20_to_30,p_30_to_40,p_more_then_40))
qchisq(.95,5)



