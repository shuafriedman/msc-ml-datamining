###sign test-- Testing a Population Median###
install.packages("DescTools")
library(DescTools)
str(t_test_data2)
data1=t_test_data3$orangesa

SignTest(data1,mu=110,alternative="less")

###paired sign test###
str(t_text_data6)
data2=t_text_data6$husband
data3=t_text_data6$wife
SignTest(data2,data3, alternative = "less",conf.level = .95)


###wilcoxon signed rank test (also used for paired tests).##
wilcox.test(data2,data3,paired=TRUE,alternative = "less",conf.level = .95)

# ###NOTE###
# the Paired Sample Sign Test and the Wilcoxon Paired
# Test are both non-parametric statistical tests that can be used to
# compare the difference between two paired samples. The Paired
# Sample Sign Test is a simple and easy-to-understand test, while the
# Wilcoxon Paired Test is a more powerful and flexible test.
# ###  ####



####wilcoxon rank sum test
# -- compare the difference between two independent samples. 
# This test is used when the data are not normally distributed or when the sample sizes are small. 
# The Wilcoxon rank sum test is often used in the same situations as
#  the t-test, which is a parametric test that assumes that the data are
#  normally distributed. However, the Wilcoxon rank sum test is less
#  sensitive to violations of this assumption, making it a more robust
#  test in certain situations.

data4=wilcoxdata$orangesa
data5=wilcoxdata$orangesb
wilcox.test(data4,data5,alternative="two.sided",conf.int = TRUE)
wilcox.test(data2,data3,alternative="less")


# Kruskal wallace test (non parametric version of Anova-- comparing means of multiple variables)
str(fdata2)
faov=aov(fdata2$km~fdata2$company)
summary(faov)
kruskal.test(fdata2$km~fdata2$company)
