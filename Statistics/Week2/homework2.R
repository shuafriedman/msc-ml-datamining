#Two Sample t test
x = t_test_data7$before
y = t_test_data7$after
t.test(x=x, y=y, alternative="two.sided") #Welch's unequal variance test is default. var.equal=False

# data:  x and y
# t = 1.0112, df = 9976, p-value = 0.3119
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -94.80996 296.86658 ---- NOTE: We can see that 0 is covered within the interval, therefore null hypothesis is Accepted.
# sample estimates:
#   mean of x mean of y 
# 4974.807  4873.779
