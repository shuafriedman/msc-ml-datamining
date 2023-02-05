# S3 method for default
# t.test(x, y = NULL,
       #alternative = c("two.sided", "less", "greater"),
       #mu = 0, paired = FALSE, var.equal = FALSE,
       #conf.level = 0.95, …)

#one sample t 
t_test_data
summary(t_test_data)
sd(t_test_data$oranges)
t.test(t_test_data$oranges,mu=100,alternative = "greater")
t_test_data2
summary(t_test_data2)
sd(t_test_data2$apples)
t.test(t_test_data2$apples,mu=100,alternative="two.sided")
# 2 sample t test
t_test_data3
summary(t_test_data3)
t.test(t_test_data3$orangesa,t_test_data3$orangesb,alternative="two.sided",var.equal=T)
summary(t_test_data4)
str(t_test_data4)
t.test(t_test_data4$applea,t_test_data4$appleb,alternative="two.sided",var.equal=T)
#paired samples
head(t_test_data5)
str(t_test_data5)
summary(t_test_data5)
t.test(t_test_data5$`pre-chanukah`,t_test_data5$`post=chanuka`,alternative="less",var.equal=T)
t.test(t_test_data5$`pre-chanukah`,t_test_data5$`post=chanuka`,alternative="less",var.equal=T,paired = TRUE)
t.test(t_text_data6$husband,t_text_data6$wife,alternative="less",var.equal=T,paired = TRUE)
